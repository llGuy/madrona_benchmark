#include "mgr.hpp"
#include "sim.hpp"
#include "import.hpp"

#include <random>
#include <numeric>
#include <algorithm>

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>
#include <madrona/physics_assets.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#include <madrona/render/asset_processor.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

#include <madrona_ktx.h>

#define MADRONA_VIEWER

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::render;
using namespace madrona::py;

namespace madEscape {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (!mgr_cfg.headlessMode) {
        if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
            return Optional<RenderGPUState>::none();
        }
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (mgr_cfg.headlessMode && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    if (!mgr_cfg.headlessMode) {
        if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
            return Optional<render::RenderManager>::none();
        }
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::maxAgents,
        .maxInstancesPerWorld = 1024,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    uint32_t raycastOutputResolution;
    bool headlessMode;

    inline Impl(const Manager::Config &mgr_cfg,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr,
                uint32_t raycast_output_resolution)
        : cfg(mgr_cfg),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr)),
          raycastOutputResolution(raycast_output_resolution),
          headlessMode(mgr_cfg.headlessMode)
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;
    MWCudaLaunchGraph renderSetupGraph;
    Optional<MWCudaLaunchGraph> renderGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec,
                   MWCudaLaunchGraph &&step_graph,
                   MWCudaLaunchGraph &&render_setup_graph,
                   Optional<MWCudaLaunchGraph> &&render_graph)
        : Impl(mgr_cfg,
               action_buffer,
               std::move(render_gpu_state), std::move(render_mgr),
               mgr_cfg.raycastOutputResolution),
          gpuExec(std::move(gpu_exec)),
          stepGraph(std::move(step_graph)),
          renderSetupGraph(std::move(render_setup_graph)),
          renderGraph(std::move(render_graph))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
        gpuExec.run(renderSetupGraph);

        if (renderGraph.has_value()) {
            gpuExec.run(*renderGraph);
        }
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#else
static_assert(false, "This only works with the CUDA backend");
#endif

static Optional<imp::SourceTexture> ktxImageImportFn(
        void *data, size_t num_bytes)
{
    ktx::ConvertedOutput converted = {};
    ktx::loadKTXMem(data, num_bytes, &converted);

    return imp::SourceTexture {
        .data = converted.textureData,
        .format = imp::SourceTextureFormat::BC7,
        .width = (uint32_t)converted.width,
        .height = (uint32_t)converted.height,
        .numBytes = converted.bufferSize
    };
}

struct LoadResult {
    std::vector<ImportedInstance> importedInstances;
    std::vector<UniqueScene> uniqueSceneInfos;
};

static imp::ImportedAssets loadScenes(
        Optional<render::RenderManager> &render_mgr,
        uint32_t first_unique_scene,
        uint32_t num_unique_scenes,
        LoadResult &load_result)
{
    const char *cache_everything = getenv("MADRONA_CACHE_ALL_BVH");
    const char *proc_thor = getenv("MADRONA_PROC_THOR");

    std::string hssd_scenes = std::filesystem::path(DATA_DIR) /
        "hssd-hab/scenes";
    std::string procthor_scenes = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR/5";
    std::string procthor_root = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thor-hab/configs";

    //Use Uncompressed because our GLTF loader doesn't support loading compressed vertex formats
    std::string procthor_obj_root = std::filesystem::path(DATA_DIR) /
        "ai2thor-hab/ai2thorhab-uncompressed/configs";

    //Uncomment this for procthor
    if (proc_thor && proc_thor[0] == '1') {
        hssd_scenes = procthor_scenes;
    }
    
    std::vector<std::string> scene_paths;

    for (const auto &dir_entry :
            std::filesystem::directory_iterator(hssd_scenes)) {
        scene_paths.push_back(dir_entry.path());
    }

    if (cache_everything && std::stoi(cache_everything) == 1) {
        num_unique_scenes = scene_paths.size();
    }

    std::vector<std::string> render_asset_paths;

    float height_offset = 0.f;
    float scale = 10.f;

    // Generate random permutation of iota
    std::vector<int> random_indices(scene_paths.size());
    std::iota(random_indices.begin(), random_indices.end(), 0);

    auto rnd_dev = std::random_device {};
    auto rng = std::default_random_engine { rnd_dev() };

    const char *seed_str = getenv("MADRONA_SEED");
    if (seed_str) {
        rng.seed(std::stoi(seed_str));
    } else {
        rng.seed(0);
    }

    std::shuffle(random_indices.begin(), random_indices.end(), rng);

    // Get all the asset paths and push unique scene infos
    uint32_t num_loaded_scenes = 0;

    for (int i = first_unique_scene; i < num_unique_scenes; ++i) {
        int random_index = random_indices[i];
        printf("Loading scene with %d\n", random_index);

        std::string scene_path = scene_paths[random_index];

        HabitatJSON::Scene loaded_scene;

        //uncomment this for procthor
        if (proc_thor && proc_thor[0] == '1') {
            loaded_scene = HabitatJSON::procThorJSONLoad(
                    procthor_root,
                    procthor_obj_root,
                    scene_path);
        } else {
            loaded_scene = HabitatJSON::habitatJSONLoad(scene_path);
        }

        // Store the current imported instances offset
        uint32_t imported_instances_offset = 
            load_result.importedInstances.size();

        UniqueScene unique_scene_info = {
            .numInstances = 0,
            .instancesOffset = imported_instances_offset,
            .center = { 0.f, 0.f, 0.f }
        };

        float stage_angle = 0;

        if (loaded_scene.stageFront[0] == -1){
            stage_angle = -pi/2;
        }

        Quat stage_rot = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                        Quat::angleAxis(stage_angle,{0,1,0});

        load_result.importedInstances.push_back({
            .position = stage_rot.rotateVec({ 0.f, 0.f, 0.f + height_offset }),
            .rotation = stage_rot,
            .scale = { scale, scale, scale },
            .objectID = (int32_t)render_asset_paths.size(),
        });

        render_asset_paths.push_back(loaded_scene.stagePath.string());

        std::unordered_map<std::string, uint32_t> loaded_gltfs;
        std::unordered_map<uint32_t, uint32_t> object_to_imported_instance;
        uint32_t num_center_contribs = 0;

        for (const HabitatJSON::AdditionalInstance &inst :
                loaded_scene.additionalInstances) {
            auto path_view = inst.gltfPath.string();
            auto extension_pos = path_view.rfind('.');
            assert(extension_pos != path_view.npos);
            auto extension = path_view.substr(extension_pos + 1);

            if (extension == "json") {
                continue;
            }

            auto [iter, insert_success] = loaded_gltfs.emplace(inst.gltfPath.string(), 
                    render_asset_paths.size());
            if (insert_success) {
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };
                
                ImportedInstance new_inst = {
                    .position = {pos.x * scale, pos.y * scale, pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }) * 
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)render_asset_paths.size(),
                };

                unique_scene_info.center += math::Vector3{
                    new_inst.position.x, new_inst.position.y, 0.f };
                num_center_contribs++;

                load_result.importedInstances.push_back(new_inst);
                render_asset_paths.push_back(inst.gltfPath.string());
            } else {
                // Push the instance to the instances array
                auto pos = Quat::angleAxis(pi_d2, { 1.f, 0.f, 0.f }).
                           rotateVec(Vector3{ inst.pos[0], inst.pos[1], 
                                              inst.pos[2] + height_offset });

                auto scale_vec = madrona::math::Diag3x3 {
                    inst.scale[0] * scale,
                    inst.scale[1] * scale,
                    inst.scale[2] * scale
                };

                ImportedInstance new_inst = {
                    .position = {pos.x * scale,pos.y * scale,pos.z * scale},
                    .rotation = Quat::angleAxis(pi_d2,{ 1.f, 0.f, 0.f }) *
                                Quat{ inst.rotation[0], inst.rotation[1],
                                      inst.rotation[2], inst.rotation[3] },
                    .scale = scale_vec,
                    .objectID = (int32_t)iter->second,
                };

                unique_scene_info.center += math::Vector3{
                    new_inst.position.x, new_inst.position.y, 0.f };
                num_center_contribs++;

                load_result.importedInstances.push_back(new_inst);
            }

            unique_scene_info.numInstances =
                load_result.importedInstances.size() - unique_scene_info.instancesOffset;
        }

        unique_scene_info.center = unique_scene_info.center / (float)num_center_contribs;

        load_result.uniqueSceneInfos.push_back(unique_scene_info);

        num_loaded_scenes++;
    }

    std::vector<const char *> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs.push_back(render_asset_paths[i].c_str());
    }

    imp::AssetImporter importer;

    // Setup importer to handle KTX images
    imp::ImageImporter &img_importer = importer.imageImporter();
    img_importer.addHandler("ktx2", ktxImageImportFn);

    std::array<char, 1024> import_err;
    auto render_assets = importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()),
        true);

    if (cache_everything && std::stoi(cache_everything) == 1) {
        exit(0);
    }

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    if (render_mgr.has_value()) {
        render_mgr->loadObjects(render_assets->objects, 
                render_assets->materials,
                render_assets->textures);

        render_mgr->configureLighting({
            { true, math::Vector3{1.0f, -1.0f, -0.05f}, math::Vector3{1.0f, 1.0f, 1.0f} }
        });
    }

    return std::move(*render_assets);
}

Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);

    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        sim_cfg.numAgents = num_agents;
    } else {
        sim_cfg.numAgents = 1;
    }

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        std::vector<ImportedInstance> imported_instances;

        sim_cfg.mergeAll = false;
        sim_cfg.dynamicMovement = mgr_cfg.dynamicMovement;

        const char *first_unique_scene_str = getenv("HSSD_FIRST_SCENE");
        const char *num_unique_scene_str = getenv("HSSD_NUM_SCENES");

        uint32_t first_scene = 0;
        uint32_t num_scenes = 1;

        if (first_unique_scene_str) {
            first_scene = std::stoi(first_unique_scene_str);
        }

        if (num_unique_scene_str) {
            num_scenes = std::stoi(num_unique_scene_str);
        }

        LoadResult load_result = {};

        auto imported_assets = loadScenes(
                render_mgr, first_scene,
                num_scenes,
                load_result);

        sim_cfg.importedInstances = (ImportedInstance *)cu::allocGPU(
                sizeof(ImportedInstance) *
                load_result.importedInstances.size());

        sim_cfg.numImportedInstances = load_result.importedInstances.size();

        sim_cfg.numUniqueScenes = load_result.uniqueSceneInfos.size();
        sim_cfg.uniqueScenes = (UniqueScene *)cu::allocGPU(
                sizeof(UniqueScene) * load_result.uniqueSceneInfos.size());

        sim_cfg.numWorlds = mgr_cfg.numWorlds;

        REQ_CUDA(cudaMemcpy(sim_cfg.importedInstances, 
                    load_result.importedInstances.data(),
                    sizeof(ImportedInstance) *
                    load_result.importedInstances.size(),
                    cudaMemcpyHostToDevice));

        REQ_CUDA(cudaMemcpy(sim_cfg.uniqueScenes, 
                    load_result.uniqueSceneInfos.data(),
                    sizeof(UniqueScene) *
                    load_result.uniqueSceneInfos.size(),
                    cudaMemcpyHostToDevice));


        if (render_mgr.has_value()) {
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        uint32_t raycast_output_resolution = mgr_cfg.raycastOutputResolution;
        CudaBatchRenderConfig::RenderMode rt_render_mode;


        // If the rasterizer is enabled, disable the raycaster
        if (mgr_cfg.enableBatchRenderer) {
            raycast_output_resolution = 0;
        } else {
            rt_render_mode = CudaBatchRenderConfig::RenderMode::RGBD;
        }


        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { HABITAT_SRC_LIST },
            { HABITAT_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx, 
        mgr_cfg.enableBatchRenderer ? Optional<madrona::CudaBatchRenderConfig>::none() : 
            madrona::CudaBatchRenderConfig {
                .renderMode = rt_render_mode,
                // .importedAssets = &imported_assets,
                .geoBVHData = render::AssetProcessor::makeBVHData(imported_assets.objects),
                .materialData = render::AssetProcessor::initMaterialData(
                        imported_assets.materials.data(), imported_assets.materials.size(),
                        imported_assets.textures.data(), imported_assets.textures.size()),
                .renderResolution = raycast_output_resolution,
                .nearPlane = 3.f,
                .farPlane = 1000.f
        });

        MWCudaLaunchGraph step_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Step);
        MWCudaLaunchGraph render_setup_graph = gpu_exec.buildLaunchGraph(
                TaskGraphID::Render);

        Optional<MWCudaLaunchGraph> render_graph = [&]() -> Optional<MWCudaLaunchGraph> {
            if (mgr_cfg.enableBatchRenderer) {
                return Optional<MWCudaLaunchGraph>::none();
            } else {
                return gpu_exec.buildRenderGraph();
            }
        } ();

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
            std::move(step_graph),
            std::move(render_setup_graph),
            std::move(render_graph)
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        FATAL("This environment doesn't support CPU backend");
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    const char *num_agents_str = getenv("HIDESEEK_NUM_AGENTS");
    if (num_agents_str) {
        uint32_t num_agents = std::stoi(num_agents_str);
        numAgents = num_agents;
    } else {
        numAgents = 1;
    }
    
    step();
}

Manager::~Manager() {}

void Manager::step()
{
    impl_->run();

    if (impl_->headlessMode) {
        if (impl_->cfg.enableBatchRenderer) {
            impl_->renderMgr->readECS();
        }
    } else {
        if (impl_->renderMgr.has_value()) {
            impl_->renderMgr->readECS();
        }
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
}

Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            numAgents,
            4,
        });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

Tensor Manager::raycastTensor() const
{
    uint32_t pixels_per_view = impl_->raycastOutputResolution *
        impl_->raycastOutputResolution;
    return impl_->exportTensor(ExportID::Raycast,
                               TensorElementType::UInt8,
                               {
                                   impl_->cfg.numWorlds*numAgents,
                                   pixels_per_view * 4,
                               });
}

void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate,
                        int32_t grab,
                        int32_t x,
                        int32_t y,
                        int32_t z,
                        int32_t rot,
                        int32_t vrot)
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        .grab = grab,
        .x = x,
        .y = y,
        .z = z,
        .rot = rot,
        .vrot = vrot,
    };

    auto *action_ptr = impl_->agentActionsBuffer +
        world_idx * numAgents + agent_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
