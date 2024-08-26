#include <string>
#include <iostream>
#include <simdjson.h>
#include <filesystem>

#include "import.hpp"

namespace HabitatJSON {

using namespace std;

std::string convertPath(std::string path){
    int index = path.find("hssd-hab");
    return path.substr(index);
}

Scene habitatJSONLoad(std::string_view scene_path_name)
{
    using namespace filesystem;
    using namespace simdjson;
    using namespace HabitatJSON;

    path scene_path(absolute(scene_path_name));
    path root_path = scene_path.parent_path().parent_path();

    path stage_dir = root_path / "stages";

    Scene scene;

    try {
        simdjson::dom::parser scene_parser;
        simdjson::dom::element root = scene_parser.load(scene_path);

        string_view stage_name = root["stage_instance"]["template_name"];

        auto stage_path = root_path / stage_name;
        stage_path.concat(".stage_config.json");

        simdjson::dom::parser stage_parser;
        auto stage_root = stage_parser.load(stage_path);
        string_view render_asset_name = stage_root["render_asset"];
        scene.stagePath = 
            stage_dir / render_asset_name;

        uint32_t front_idx = 0;
        auto front_obj = stage_root["front"];
        for (auto c : front_obj) {
            scene.stageFront[front_idx++] = float(double(c));
        }

#if 0
        string_view lighting_path_str = string_view(root["default_lighting"]);
        if (lighting_path_str.length() > 0) {
            path lighting_path = lighting_path_str;
            lighting_path.concat(".lighting_config.json");

            simdjson::dom::parser light_parser;
            auto light_root = light_parser.load(root_path / lighting_path);

            vector<Light> lights;
            for (auto [idx, light] : dom::object(light_root["lights"])) {
                string_view type_str = light["type"];
                LightType type;
                if (type_str == "point") {
                    type = LightType::Point;
                } else if (type_str == "environment") {
                    type = LightType::Environment;
                } else {
                    cerr << scene_path_name << ": Unknown light type" << endl;
                    abort();
                }

                if (type == LightType::Point) {
                    uint32_t vec_idx = 0;

                    Light dst_light;
                    dst_light.type = type;

                    for (auto c : light["position"]) {
                        dst_light.position[vec_idx++] = float(double(c));
                    }

                    dst_light.intensity = double(light["intensity"]);
                    if (dst_light.intensity <= 0.f) {
                        cerr << "Warning: Skipping negative intensity light" << endl;
                        continue;
                    }

                    vec_idx = 0;
                    for (auto c : light["color"]) {
                        dst_light.color[vec_idx++] = float(double(c));
                    }

                    scene.lights.push_back(dst_light);
                } else if (type == LightType::Environment) {
                    // ignore environment map
                }
            }
        }
#endif

        simdjson::dom::parser nested_parser;
        auto insts = root["object_instances"];

        for (const auto &inst : insts) {
            AdditionalInstance additional_inst;

            uint32_t idx = 0;
            auto translation_obj = inst["translation"];
            for (auto c : translation_obj) {
                additional_inst.pos[idx++] = float(double(c));
            }

            // Some don't have rotation
            idx = 0;
            simdjson::dom::array rotation_obj;
            auto res = inst.at_key("rotation").get(rotation_obj);

            if (!res) {
                for (auto c : rotation_obj) {
                    additional_inst.rotation[idx++] = float(double(c));
                } 
            } else {
                additional_inst.rotation[0] = 1;
                additional_inst.rotation[1] = 0;
                additional_inst.rotation[2] = 0;
                additional_inst.rotation[3] = 0;
            }

            idx = 0;
            simdjson::dom::array scale;
            res = inst.at_key("non_uniform_scale").get(scale);

            if (!res) {
                for (auto c : scale) {
                    additional_inst.scale[idx++] = float(double(c));
                }
            } else {
                for(int i = 0; i < 3; i++){
                    additional_inst.scale[i] = 1;
                }
            }

            string_view template_name = inst["template_name"];

            path object_glb_path = root_path / "objects";
            path object_config_path = root_path / "objects";

            bool is_decomposed = false;

            if (template_name.length() > 10) {
                auto part_start = template_name.find("part");

                if (part_start != std::string::npos) {
                    is_decomposed = true;
                    string core_name = string(template_name);
                    core_name.resize(part_start - 1);
                    object_glb_path = object_glb_path / "decomposed";
                    object_glb_path = object_glb_path / core_name;
                    object_glb_path = object_glb_path / template_name;
                    object_glb_path.concat(".glb");

                    object_config_path = object_config_path / string(1, template_name[0]);
                    object_config_path = object_config_path / core_name;
                    object_config_path.concat(".object_config.json");
                }
                else {
                    object_config_path = object_config_path / string(1, template_name[0]);
                    object_config_path = object_config_path / template_name;
                    object_config_path.concat(".object_config.json");

                    object_glb_path = object_glb_path / string(1, template_name[0]);
                    object_glb_path = object_glb_path / template_name;
                    object_glb_path.concat(".glb");
                }
            } else {
                object_config_path = object_config_path / "openings";
                object_config_path = object_config_path / template_name;
                object_config_path.concat(".object_config.json");

                object_glb_path = object_glb_path / "openings";
                object_glb_path = object_glb_path / template_name;
                object_glb_path.concat(".object_config.json");
            }

            auto inst_root = nested_parser.load(object_config_path);
            string_view inst_asset = inst_root["render_asset"];

            additional_inst.name = string(template_name);
            additional_inst.gltfPath = object_glb_path;
            additional_inst.dynamic =
                string_view(inst["motion_type"]) == "DYNAMIC";

            scene.additionalInstances.push_back(additional_inst);
        }

        simdjson::dom::array objs;
        auto obj_err = root["additional_objects"].get(objs);
        if (!obj_err) {
            for (const auto &obj : objs) {
                string_view template_name = obj["template_name"];
                auto template_path = root_path / template_name;
                template_path.concat(".object_config.json");
                auto obj_root = nested_parser.load(template_path);
                string_view obj_asset = obj_root["render_asset"];

                auto obj_path = template_path.parent_path() / obj_asset;
                scene.additionalObjects.push_back({
                    string(obj["name"]),
                    obj_path,
                });
            }
        }
    } catch (const simdjson_error &e) {
        cerr << "Habitat JSON loading '" << scene_path_name
             << "' failed: " << e.what() << endl;
        abort();
    };

    return scene;
}

Scene procThorJSONLoad(std::string_view root_paths,
                       std::string_view obj_root_paths,
                       std::string_view scene_path_name)
{
    using namespace filesystem;
    using namespace simdjson;
    using namespace HabitatJSON;

    path scene_path(absolute(scene_path_name));
    path root_path = root_paths;

    path stage_dir = root_path / "stages";

    Scene scene;

    try {
        simdjson::dom::parser scene_parser;
        simdjson::dom::element root = scene_parser.load(scene_path);

        string_view stage_name = root["stage_instance"]["template_name"];
        auto stage_path = root_path / stage_name;
        stage_path.concat(".stage_config.json");

        simdjson::dom::parser stage_parser;
        auto stage_root = stage_parser.load(stage_path);
        string_view render_asset_name = stage_root["render_asset"];
        scene.stagePath =
            stage_path.parent_path() / render_asset_name;

        uint32_t front_idx = 0;
        auto front_obj = stage_root["front"];
        for (auto c : front_obj) {
            scene.stageFront[front_idx++] = float(double(c));
        }

        string_view lighting_path_str;

        if(!root["default_lighting"].get(lighting_path_str)){
            lighting_path_str = "";
        }

        if (lighting_path_str.length() > 0) {
            path lighting_path = lighting_path_str;
            lighting_path.concat(".lighting_config.json");

            simdjson::dom::parser light_parser;
            auto light_root = light_parser.load(root_path / lighting_path);

            vector<Light> lights;
            for (auto [idx, light] : dom::object(light_root["lights"])) {
                string_view type_str = light["type"];
                LightType type;
                if (type_str == "point") {
                    type = LightType::Point;
                } else if (type_str == "environment") {
                    type = LightType::Environment;
                } else {
                    cerr << scene_path_name << ": Unknown light type" << endl;
                    abort();
                }

                if (type == LightType::Point) {
                    uint32_t vec_idx = 0;

                    Light dst_light;
                    dst_light.type = type;

                    for (auto c : light["position"]) {
                        dst_light.position[vec_idx++] = float(double(c));
                    }

                    dst_light.intensity = double(light["intensity"]);
                    if (dst_light.intensity <= 0.f) {
                        cerr << "Warning: Skipping negative intensity light" << endl;
                        continue;
                    }

                    vec_idx = 0;
                    for (auto c : light["color"]) {
                        dst_light.color[vec_idx++] = float(double(c));
                    }

                    scene.lights.push_back(dst_light);
                } else if (type == LightType::Environment) {
                    // ignore environment map
                }
            }
        }

        simdjson::dom::parser nested_parser;
        auto insts = root["object_instances"];

        for (const auto &inst : insts) {
            AdditionalInstance additional_inst;

            uint32_t idx = 0;
            auto translation_obj = inst["translation"];
            for (auto c : translation_obj) {
                additional_inst.pos[idx++] = float(double(c));
            }

            // Some don't have rotation
            idx = 0;
            simdjson::dom::array rotation_obj;
            auto res = inst.at_key("rotation").get(rotation_obj);

            if (!res) {
                for (auto c : rotation_obj) {
                    additional_inst.rotation[idx++] = float(double(c));
                }
            } else {
                additional_inst.rotation[0] = 1;
                additional_inst.rotation[1] = 0;
                additional_inst.rotation[2] = 0;
                additional_inst.rotation[3] = 0;
            }

            idx = 0;
            simdjson::dom::array scale;
            res = inst.at_key("non_uniform_scale").get(scale);

            if (!res) {
                for (auto c : scale) {
                    additional_inst.scale[idx++] = float(double(c));
                }
            } else {
                for(int i = 0; i < 3; i++){
                    additional_inst.scale[i] = 1;
                }
            }

            string_view template_name = inst["template_name"];

            path object_glb_path = path(obj_root_paths) / "objects";
            path object_config_path = obj_root_paths;
            object_config_path = object_config_path / template_name;
            object_config_path.concat(".object_config.json");

            auto inst_root = nested_parser.load(object_config_path);
            string_view inst_asset = inst_root["render_asset"];
            object_glb_path = object_glb_path / inst_asset;

            additional_inst.name = string(template_name);
            additional_inst.gltfPath = object_glb_path;
            additional_inst.dynamic =
                string_view(inst["motion_type"]) == "DYNAMIC";

            scene.additionalInstances.push_back(additional_inst);
        }

    } catch (const simdjson_error &e) {
        cerr << "Habitat JSON loading '" << scene_path_name
             << "' failed: " << e.what() << endl;
        abort();
    };

    return scene;
}

}
