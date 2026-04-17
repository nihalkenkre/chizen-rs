use std::ffi::CStr;

use ash::vk::{self};

use nalgebra_glm as glm;

use crate::{
    viewport,
    vulkan_objects::{self, AllocatorTrait},
};

// for triangles meshes
macro_rules! MESH_MASK {
    () => {
        0x1 << 0
    };
}

// for punctual lights procedural geometry
macro_rules! LIGHT_MASK {
    () => {
        0x1 << 1
    };
}

#[derive(Default)]
pub enum SceneType {
    World(Scene),
    #[default]
    None,
}

pub trait SceneTrait {
    fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        debug_utils: &vulkan_objects::DebugUtils,
    );

    fn update_final_render_target_desc(
        &self,
        device: &ash::Device,
        final_render_target_desc: vk::DescriptorImageInfo,
    );

    fn update_accum_target_desc(
        &self,
        device: &ash::Device,
        accum_target_desc: vk::DescriptorImageInfo,
    );

    fn render(
        &self,
        device: &ash::Device,
        random_states: vk::DeviceAddress,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        render_extent: vk::Extent2D,
        command_buffer: vk::CommandBuffer,
        current_sample: u32,
        render_mode: u32,
        camera_index: u32,
        bounce_count: u32,
        common_scene_data: &viewport::CommonSceneData,
    );

    fn destroy(&mut self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType);
}

impl SceneTrait for SceneType {
    fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        match self {
            SceneType::World(scene) => {
                scene.reload_shaders(
                    device,
                    queue,
                    accum_target,
                    final_render_target,
                    debug_utils,
                );
            }
            SceneType::None => {}
        }
    }

    fn update_accum_target_desc(
        &self,
        device: &ash::Device,
        accum_target_desc: vk::DescriptorImageInfo,
    ) {
        match self {
            SceneType::World(scene) => {
                scene.update_accum_target_desc(device, accum_target_desc);
            }
            SceneType::None => {}
        }
    }

    fn update_final_render_target_desc(
        &self,
        device: &ash::Device,
        final_render_target_desc: vk::DescriptorImageInfo,
    ) {
        match self {
            SceneType::World(scene) => {
                scene.update_final_render_target_desc(device, final_render_target_desc);
            }
            SceneType::None => {}
        }
    }

    fn render(
        &self,
        device: &ash::Device,
        random_states: vk::DeviceAddress,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        render_extent: vk::Extent2D,
        command_buffer: vk::CommandBuffer,
        current_sample: u32,
        render_mode: u32,
        camera_index: u32,
        bounce_count: u32,
        common_scene_data: &viewport::CommonSceneData,
    ) {
        match self {
            SceneType::World(scene) => {
                scene.render(
                    device,
                    random_states,
                    accum_target,
                    final_render_target,
                    render_extent,
                    command_buffer,
                    current_sample,
                    render_mode,
                    camera_index,
                    bounce_count,
                    common_scene_data,
                );
            }
            SceneType::None => {}
        }
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        match self {
            SceneType::World(scene) => scene.destroy(device, allocator),
            SceneType::None => {}
        }
    }
}

#[derive(Debug, Default)]
struct PipelineData {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,

    pub groups_count: u32,
    pub miss_groups_count: u32,
    pub hit_groups_count: u32,
}

#[derive(Debug, Default, Clone, Copy)]
struct PushConstants {
    camera_instances: vk::DeviceAddress,
    punctual_lights: vk::DeviceAddress,
    mesh_emitters: vk::DeviceAddress,
    materials: vk::DeviceAddress,

    random_states: vk::DeviceAddress,
    current_sample: u32,

    render_mode: u32,
    camera_index: u32,
    punctual_lights_count: u32,
    mesh_emitters_count: u32,
    bounce_count: u32,
}

unsafe impl bytemuck::NoUninit for PushConstants {}

impl PipelineData {
    pub fn new(
        device: &ash::Device,
        debug_utils: &vulkan_objects::DebugUtils,
        textures_count: u32,
        raytracing_device: &ash::khr::ray_tracing_pipeline::Device,
    ) -> Self {
        let spirv_data = std::fs::read(
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
                + ("/shaders/render.slang.spv"),
        )
        .expect("Could not read render.slang.spv");

        let shader_mod;
        unsafe {
            shader_mod = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(spirv_data.align_to::<u32>().1),
                    None,
                )
                .unwrap();
        }

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("raygen\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("mesh_miss\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("light_miss\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::ANY_HIT_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("mesh_anyhit\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("mesh_closesthit\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::INTERSECTION_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("light_intersection\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("light_closesthit\0".as_bytes()).unwrap()),
        ];

        let groups = [
            // raygen
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR),
            // mesh miss
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR),
            // light miss
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(2)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR),
            // mesh triangle hit group
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(3)
                .closest_hit_shader(4),
            // light procedural hit group
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(5)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(6),
        ];

        let mut miss_groups_count = 0;
        let mut hit_groups_count = 0;

        for g in 1..groups.len() {
            // 0 is raygen group, so skip it
            let group = groups[g];
            if group.general_shader != vk::SHADER_UNUSED_KHR {
                miss_groups_count += 1;
            }

            if group.any_hit_shader != vk::SHADER_UNUSED_KHR
                || group.closest_hit_shader != vk::SHADER_UNUSED_KHR
                || group.intersection_shader != vk::SHADER_UNUSED_KHR
            {
                hit_groups_count += 1;
            }
        }

        let dsl_0_binds = [
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .binding(0)
                .descriptor_count(1),
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                .binding(1)
                .descriptor_count(1),
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR
                        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                        | vk::ShaderStageFlags::MISS_KHR
                        | vk::ShaderStageFlags::ANY_HIT_KHR
                        | vk::ShaderStageFlags::INTERSECTION_KHR,
                )
                .binding(2)
                .descriptor_count(1),
            vk::DescriptorSetLayoutBinding::default()
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .stage_flags(
                    vk::ShaderStageFlags::CLOSEST_HIT_KHR | vk::ShaderStageFlags::ANY_HIT_KHR,
                )
                .binding(3)
                .descriptor_count(textures_count),
        ];

        let binding_flags = [
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::empty(),
            vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        ];

        let mut dsl_0_binds_flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .push_next(&mut dsl_0_binds_flags_ci)
            .bindings(&dsl_0_binds);

        let descriptor_set_layout;

        unsafe {
            descriptor_set_layout = device.create_descriptor_set_layout(&dsl_ci, None).unwrap();
        }

        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR
                    | vk::ShaderStageFlags::INTERSECTION_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
            )
            .size(size_of::<PushConstants>() as u32)];

        let dsls = [descriptor_set_layout];
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&pc_ranges)
            .set_layouts(&dsls);

        let layout;
        unsafe {
            layout = device.create_pipeline_layout(&layout_ci, None).unwrap();
        }

        let cis = [vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stages)
            .groups(&groups)
            .max_pipeline_ray_recursion_depth(1)
            .layout(layout)];

        let pipeline;
        unsafe {
            pipeline = raytracing_device
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &cis,
                    None,
                )
                .unwrap()[0];

            device.destroy_shader_module(shader_mod, None);
        }

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(pipeline, "renderer separate scene pipeline");
            debug_utils.set_object_name(layout, "renderer separate scene pipeline layout");
            debug_utils.set_object_name(descriptor_set_layout, "renderer separate scene dsl");
        }

        Self {
            pipeline,
            layout,
            descriptor_set_layout,
            groups_count: groups.len() as u32,
            miss_groups_count,
            hit_groups_count,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct Scene {
    pipeline_data: PipelineData,

    descriptor_set: vk::DescriptorSet,
    descriptor_pool: vk::DescriptorPool,

    // storing raw buffer and allocation_type since BufferResource cannot be safely across threads since it contains a *mut c_void for mapped_ptr, which does not implement Sync
    rg_sbt: vk::Buffer,
    rg_sbt_allocation: vulkan_objects::AllocationType,
    ms_sbt: vk::Buffer,
    ms_sbt_allocation: vulkan_objects::AllocationType,
    hg_sbt: vk::Buffer,
    hg_sbt_allocation: vulkan_objects::AllocationType,

    rg_region: vk::StridedDeviceAddressRegionKHR,
    ms_region: vk::StridedDeviceAddressRegionKHR,
    hg_region: vk::StridedDeviceAddressRegionKHR,

    positions: vk::Buffer,
    positions_allocation: vulkan_objects::AllocationType,
    indices: vk::Buffer,
    indices_allocation: vulkan_objects::AllocationType,
    vertices_data: vk::Buffer,
    vertices_data_allocation: vulkan_objects::AllocationType,
    light_bboxes: vk::Buffer,
    light_bboxes_allocation: vulkan_objects::AllocationType,
    blases: Vec<vulkan_objects::BLAS>,
    tlas: vulkan_objects::TLAS,
    accel_struct_device: ash::khr::acceleration_structure::Device,
    raytracing_device: ash::khr::ray_tracing_pipeline::Device,

    textures_count: u32,
    common_scene_data: viewport::CommonSceneData,
}

impl Scene {
    pub fn new(
        gltf: &gltf::Document,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
        instance: &ash::Instance,
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        queue_family_indices: &[u32],
        compute_queue: vk::Queue,
        raytracing_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        raytracing_loader: &ash::khr::ray_tracing_pipeline::Device,
        scratch_buffer_alignment: u32,
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
        compute_helper: &mut vulkan_objects::DataHelper,
        common_scene_data: &viewport::CommonSceneData,
    ) -> Self {
        let pipeline_data = PipelineData::new(
            device,
            debug_utils,
            common_scene_data.images_descriptor_info.len() as u32,
            raytracing_loader,
        );

        // All data is used to create the BLAS and instance for TLAS
        #[derive(Debug, Default)]
        struct Primitive {
            pub positions_offset: vk::DeviceSize,
            pub vertices_data_offset: vk::DeviceSize,
            pub indices_offset: vk::DeviceSize,
            pub material_index: u32,
            pub vertices_count: u32,
            pub indices_count: u32,
            pub model_matrix: glm::Mat4,
            pub is_emissive: bool,
        }

        #[derive(Debug, Default)]
        struct Mesh {
            pub primitives: Vec<Primitive>,
        }

        #[derive(Debug, Default)]
        struct LightInfo {
            position: glm::Vec3,
            bbox_offset: vk::DeviceSize,
        }

        let mut positions = vec![];
        let mut vertices_data = vec![];
        let mut indices = vec![];

        let mut meshes = vec![];

        let mut light_bboxes = vec![];
        // the offset into the light bboxes array, passed during the BLAS and instance for TLAS. light_bboxes.device_or_host_address + offset
        let mut light_infos = vec![];

        // run through the nodes and get the mesh and light data
        // other "common_scene_data" are got from the viewport scene
        for node in gltf.nodes() {
            if let Some(mesh) = node.mesh() {
                let mut primitives = vec![];

                for curr_prim in mesh.primitives() {
                    let material_index = match curr_prim.material().index() {
                        Some(index) => (index + 1) as u32, // + 1 since index 0 is the default material
                        None => 0,
                    };

                    let reader = curr_prim.reader(|buffer| Some(&buffers[buffer.index()]));

                    let mut prim_positions =
                        reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();

                    let mut tangents = vec![];
                    // if tangents not found create an array with zeros, required for VertexData struct
                    if let Some(t) = reader.read_tangents() {
                        tangents = t.collect::<Vec<[f32; 4]>>();
                    } else {
                        eprintln!(
                            "No tangents for {:?}",
                            mesh.name().unwrap_or("default mesh")
                        );

                        tangents = vec![[0f32, 0f32, 0f32, 0f32]; prim_positions.len()];
                    };

                    let normals = reader.read_normals().unwrap().collect::<Vec<[f32; 3]>>();

                    let mut uvs_0 = vec![];
                    // if uvs not found create an array with zeros, required for VertexData struct
                    if let Some(uv) = reader.read_tex_coords(0) {
                        uvs_0 = uv.into_f32().collect::<Vec<[f32; 2]>>()
                    } else {
                        println!("No uvs_0 for {:?}", mesh.name().unwrap_or("default mesh"));

                        uvs_0 = vec![[0f32, 0f32]; prim_positions.len()];
                    }

                    let mut prim_vertices_data = vec![];
                    for v in 0..prim_positions.len() {
                        let vertex_data = viewport::VertexData {
                            tangent: tangents[v],
                            normal: [normals[v][0], normals[v][1], normals[v][2], 0f32],
                            uv_0: uvs_0[v],
                            ..Default::default()
                        };
                        prim_vertices_data.push(vertex_data);
                    }

                    let mut prim_indices = reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .collect::<Vec<u32>>();

                    // alignement required to create the BLAS
                    indices.resize(aligned_size!(indices.len(), size_of::<u32>()), 0);

                    // alignement required to create the BLAS
                    positions.resize(
                        aligned_size!(positions.len(), size_of::<[f32; 3]>()),
                        [0f32; 3],
                    );

                    // get the average emission for the mesh, if more than 0 then the mesh will be added to the mesh emitter list
                    let avg_emission: [f32; 3] = match curr_prim.material().emissive_texture() {
                        Some(texture) => {
                            let mut total_sum: u32 = 0;

                            let image = images.get(texture.texture().source().index()).unwrap();

                            let avg_emission: [f32; 3] = match image.format {
                                gltf::image::Format::R8G8B8A8 => {
                                    let mut pixels_u32: Vec<u32> = vec![];

                                    for pixel in &image.pixels {
                                        pixels_u32.push(*pixel as u32);
                                    }

                                    let pixels =
                                        unsafe { pixels_u32.align_to::<[u32; 4]>().1.to_vec() };

                                    let pixels_len = pixels.len() as u32;
                                    let sum_pixels = pixels
                                        .into_iter()
                                        .reduce(|acc, e| {
                                            [
                                                acc[0] + e[0],
                                                acc[1] + e[1],
                                                acc[2] + e[2],
                                                acc[3] + e[3],
                                            ]
                                        })
                                        .unwrap();

                                    [
                                        (sum_pixels[0] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[0]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                        (sum_pixels[1] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[1]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                        (sum_pixels[2] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[2]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                    ]
                                }
                                gltf::image::Format::R8G8B8 => {
                                    let mut pixels_u32: Vec<u32> = vec![];

                                    for pixel in &image.pixels {
                                        pixels_u32.push(*pixel as u32);
                                    }

                                    let pixels =
                                        unsafe { pixels_u32.align_to::<[u32; 3]>().1.to_vec() };

                                    let pixels_len = pixels.len() as u32;
                                    let sum_pixels = pixels
                                        .into_iter()
                                        .reduce(|acc, e| {
                                            [acc[0] + e[0], acc[1] + e[1], acc[2] + e[2]]
                                        })
                                        .unwrap();

                                    [
                                        (sum_pixels[0] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[0]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                        (sum_pixels[1] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[1]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                        (sum_pixels[2] / pixels_len) as f32 / 255f32
                                            * curr_prim.material().emissive_factor()[2]
                                            * curr_prim
                                                .material()
                                                .emissive_strength()
                                                .unwrap_or(1f32),
                                    ]
                                }
                                _ => [0f32, 0f32, 0f32],
                            };

                            [
                                curr_prim.material().emissive_factor()[0]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                                curr_prim.material().emissive_factor()[1]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                                curr_prim.material().emissive_factor()[2]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                            ]
                        }
                        None => {
                            let emission = [
                                curr_prim.material().emissive_factor()[0]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                                curr_prim.material().emissive_factor()[1]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                                curr_prim.material().emissive_factor()[2]
                                    * curr_prim.material().emissive_strength().unwrap_or(1f32),
                            ];

                            emission
                        }
                    };

                    let mut is_emissive = false;

                    if avg_emission[0] > 0f32 || avg_emission[1] > 0f32 || avg_emission[1] > 0f32 {
                        is_emissive = true;
                    }

                    primitives.push(Primitive {
                        positions_offset: (positions.len() * size_of::<[f32; 3]>())
                            as vk::DeviceSize,
                        vertices_data_offset: (vertices_data.len()
                            * size_of::<viewport::VertexData>())
                            as vk::DeviceSize,
                        indices_offset: (indices.len() * size_of::<u32>()) as vk::DeviceSize,
                        material_index,
                        vertices_count: prim_positions.len() as u32,
                        indices_count: prim_indices.len() as u32,
                        model_matrix: glm::make_mat4(node.transform().matrix().as_flattened()),
                        is_emissive,
                    });

                    indices.append(&mut prim_indices);
                    positions.append(&mut prim_positions);
                    vertices_data.append(&mut prim_vertices_data);
                }

                meshes.push(Mesh { primitives });
            } else if let Some(light) = node.light() {
                let radius = match light.extras() {
                    Some(extras) => {
                        let v: serde_json::Value = serde_json::from_str(extras.get()).unwrap();

                        match v {
                            serde_json::Value::Object(v) => {
                                v.get("radius")
                                    .unwrap_or(&serde_json::to_value(0.001f32).unwrap())
                                    .as_f64()
                                    .unwrap()
                                    .max(0.001) as f32
                            }
                            _ => 0.001,
                        }
                    }
                    None => 0.001,
                };

                let position = glm::make_vec3(&node.transform().decomposed().0);

                light_infos.push(LightInfo {
                    position,
                    bbox_offset: (light_bboxes.len() * size_of::<vk::AabbPositionsKHR>())
                        as vk::DeviceSize,
                });

                light_bboxes.push(
                    vk::AabbPositionsKHR::default()
                        .min_x(-radius)
                        .min_y(-radius)
                        .min_z(-radius)
                        .max_x(radius)
                        .max_y(radius)
                        .max_z(radius),
                );
            }
        }

        // prepare to upload the data to the device
        let staging_positions_resource = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &positions.as_flattened(),
            debug_utils,
            None,
            "staging positions",
        );

        let positions_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            (positions.len() * size_of_val(&positions[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "positions",
        );

        let staging_indices_resource = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &indices,
            debug_utils,
            None,
            "staging indices",
        );

        let indices_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            (indices.len() * size_of_val(&indices[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "indices",
        );

        let staging_vertices_data_resource = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &vertices_data,
            debug_utils,
            None,
            "staging vertices data",
        );

        let vertices_data_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (vertices_data.len() * size_of_val(&vertices_data[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "vertices data",
        );
        let mut staging_light_bboxes_resource = vulkan_objects::BufferResource::default();
        let mut light_bboxes_resource = vulkan_objects::BufferResource::default();

        if light_infos.len() > 0 {
            staging_light_bboxes_resource = allocator.create_buffer_on_host_with_data(
                device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &light_bboxes,
                debug_utils,
                None,
                "staging light bboxes",
            );

            light_bboxes_resource = allocator.create_buffer_on_device(
                device,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                (light_bboxes.len() * size_of_val(&light_bboxes[0])) as vk::DeviceSize,
                debug_utils,
                None,
                "light bboxes",
            );
        }

        data_helper.record_batch(device);

        data_helper.copy_buffer_to_buffer(
            device,
            staging_positions_resource.vk_buffer(),
            positions_resource.vk_buffer(),
            staging_positions_resource.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_indices_resource.vk_buffer(),
            indices_resource.vk_buffer(),
            staging_indices_resource.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_vertices_data_resource.vk_buffer(),
            vertices_data_resource.vk_buffer(),
            staging_vertices_data_resource.data_size,
        );

        if light_infos.len() > 0 {
            data_helper.copy_buffer_to_buffer(
                device,
                staging_light_bboxes_resource.vk_buffer(),
                light_bboxes_resource.vk_buffer(),
                staging_light_bboxes_resource.data_size,
            );
        }

        data_helper.submit_batch(device, None, None, None);

        allocator.destroy_buffer_resource(device, staging_positions_resource);
        allocator.destroy_buffer_resource(device, staging_indices_resource);
        allocator.destroy_buffer_resource(device, staging_vertices_data_resource);
        allocator.destroy_buffer_resource(device, staging_light_bboxes_resource);

        // create the ASes and shader binding tables for the trace ray command
        #[derive(Debug, Clone, Copy)]
        struct HGRecordData {
            vertices_data: vk::DeviceAddress,
            indices: vk::DeviceAddress,
            material_index: u32,
            light_index: u32,
            mesh_emitter_index: i32,
        }

        unsafe impl bytemuck::NoUninit for HGRecordData {}

        let raytracing_device = ash::khr::ray_tracing_pipeline::Device::new(instance, device);
        let accel_struct_device = ash::khr::acceleration_structure::Device::new(instance, device);

        let mut instances = vec![];
        let mut blases = vec![];

        let handle_size = raytracing_properties.shader_group_handle_size;
        let aligned_handle_size = aligned_size!(
            handle_size,
            raytracing_properties.shader_group_handle_alignment
        );

        let groups_handles_size = (pipeline_data.groups_count * handle_size) as usize;

        let group_handles;
        unsafe {
            group_handles = raytracing_loader
                .get_ray_tracing_shader_group_handles(
                    pipeline_data.pipeline,
                    0,
                    pipeline_data.groups_count,
                    groups_handles_size,
                )
                .unwrap();
        }

        // create the raygen group sbt
        let mut rg_group_handle = group_handles[0..handle_size as usize].to_vec();
        rg_group_handle.resize(aligned_handle_size as usize, 0);

        let staging_rg_sbt = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &rg_group_handle,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "staging rg sbt",
        );

        // prepare to upload raygen group sbt to the device
        let rg_sbt = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            (rg_group_handle.len() * size_of_val(&rg_group_handle[0])) as vk::DeviceSize,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "rg sbt",
        );

        // create the miss group sbt
        let mut ms_group_handles = vec![];
        for m in 0..pipeline_data.miss_groups_count {
            let start_offset = (handle_size + handle_size * m) as usize;
            let mut header =
                group_handles[start_offset..start_offset + handle_size as usize].to_vec();
            ms_group_handles.append(&mut header);
        }

        // alignment necessary
        ms_group_handles.resize(
            aligned_size!(
                ms_group_handles.len(),
                raytracing_properties.shader_group_handle_alignment as usize
            ),
            0,
        );

        // prepare to upload miss group sbt to device
        let staging_ms_sbt = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &ms_group_handles,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "staging ms sbt",
        );

        let ms_sbt = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            (ms_group_handles.len() * size_of_val(&ms_group_handles[0])) as vk::DeviceSize,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "ms sbt",
        );

        let mut hit_group_handles = vec![];
        let mut mesh_emitter_count = -1;

        // for each mesh/prim, create the BLAS, populate the hitgroups with record dataa, and create the instance to be used for the TLAS
        for curr_mesh in meshes {
            for curr_prim in curr_mesh.primitives {
                // create BLAS
                let mut positions_addr = vk::DeviceOrHostAddressConstKHR::default();
                positions_addr.device_address =
                    positions_resource.device_address + curr_prim.positions_offset;

                let mut indices_addr = vk::DeviceOrHostAddressConstKHR::default();
                indices_addr.device_address =
                    indices_resource.device_address + curr_prim.indices_offset;

                let blas = vulkan_objects::BLAS::new_from_mesh(
                    device,
                    allocator,
                    &accel_struct_device,
                    positions_addr,
                    indices_addr,
                    curr_prim.vertices_count,
                    curr_prim.indices_count,
                    scratch_buffer_alignment as vk::DeviceSize,
                    compute_helper,
                    debug_utils,
                    "some prim",
                );

                // populate hit groups
                let mut mesh_emitter_index = -1;
                if curr_prim.is_emissive {
                    mesh_emitter_count += 1;
                    mesh_emitter_index = mesh_emitter_count;
                }

                // populate data for hitgroup for each ray type
                for hg in 0..pipeline_data.hit_groups_count {
                    let start_offset = (handle_size
                        + (handle_size * pipeline_data.miss_groups_count)
                        + (handle_size * hg)) as usize;
                    let mut handle =
                        group_handles[start_offset..start_offset + handle_size as usize].to_vec();

                    let sbt_record = HGRecordData {
                        indices: indices_resource.device_address + curr_prim.indices_offset,
                        vertices_data: vertices_data_resource.device_address
                            + curr_prim.vertices_data_offset,
                        material_index: curr_prim.material_index,
                        light_index: 0,
                        mesh_emitter_index,
                    };

                    handle.append(&mut bytemuck::pod_collect_to_vec(&bytemuck::bytes_of(
                        &sbt_record,
                    )));

                    handle.resize(
                        aligned_size!(
                            handle.len(),
                            raytracing_properties.shader_group_handle_alignment as usize
                        ),
                        0,
                    );

                    hit_group_handles.append(&mut handle);
                }

                // create instance for TLAS
                let accel_struct_device_addr;
                unsafe {
                    accel_struct_device_addr = accel_struct_device
                        .get_acceleration_structure_device_address(
                            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                                .acceleration_structure(blas.accel_struct),
                        );
                }

                let mut transform = vk::TransformMatrixKHR { matrix: [0f32; 12] };

                unsafe {
                    std::ptr::copy(
                        glm::transpose(&curr_prim.model_matrix).as_ptr(),
                        transform.matrix.as_mut_ptr(),
                        12,
                    );
                }

                let instance = vk::AccelerationStructureInstanceKHR {
                    acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                        device_handle: accel_struct_device_addr,
                    },
                    instance_custom_index_and_mask: vk::Packed24_8::new(0, MESH_MASK!()),
                    instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                        blases.len() as u32 * pipeline_data.hit_groups_count,
                        0,
                    ),
                    transform,
                };

                instances.push(instance);
                blases.push(blas);
            }
        }

        // for each light, create the BLAS, populate the hitgroups with record dataa, and create the instance to be used for the TLAS
        for (l, curr_light) in light_infos.iter().enumerate() {
            // create BLAS
            let mut bbox_addr = vk::DeviceOrHostAddressConstKHR::default();
            bbox_addr.device_address =
                light_bboxes_resource.device_address + curr_light.bbox_offset;

            let blas = vulkan_objects::BLAS::new_from_proc(
                device,
                allocator,
                &accel_struct_device,
                bbox_addr,
                scratch_buffer_alignment as vk::DeviceSize,
                compute_helper,
                debug_utils,
                "some light",
            );

            // populate hit groups
            for hg in 0..pipeline_data.hit_groups_count {
                let start_offset = (handle_size
                    + (handle_size * pipeline_data.miss_groups_count)
                    + (handle_size * hg)) as usize;
                let mut handle =
                    group_handles[start_offset..start_offset + handle_size as usize].to_vec();

                let sbt_record = HGRecordData {
                    indices: 0,
                    material_index: 0,
                    vertices_data: 0,
                    light_index: l as u32,
                    mesh_emitter_index: 0,
                };

                handle.append(&mut bytemuck::pod_collect_to_vec(&bytemuck::bytes_of(
                    &sbt_record,
                )));

                handle.resize(
                    aligned_size!(
                        handle.len(),
                        raytracing_properties.shader_group_handle_alignment as usize
                    ),
                    0,
                );

                hit_group_handles.append(&mut handle);
            }

            // create instance for TLAS
            let accel_struct_device_addr;
            unsafe {
                accel_struct_device_addr = accel_struct_device
                    .get_acceleration_structure_device_address(
                        &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                            .acceleration_structure(blas.accel_struct),
                    );
            }

            let mut transform = vk::TransformMatrixKHR { matrix: [0f32; 12] };

            unsafe {
                std::ptr::copy(
                    glm::transpose(&glm::translate(&glm::identity(), &curr_light.position))
                        .as_ptr(),
                    transform.matrix.as_mut_ptr(),
                    12,
                );
            }

            let instance = vk::AccelerationStructureInstanceKHR {
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: accel_struct_device_addr,
                },
                instance_custom_index_and_mask: vk::Packed24_8::new(0, LIGHT_MASK!()),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    blases.len() as u32 * pipeline_data.hit_groups_count,
                    0,
                ),
                transform,
            };

            instances.push(instance);
            blases.push(blas);
        }

        // prepare to upload the hitgroups sbt to the device
        let staging_hg_sbt = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &hit_group_handles,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "staging hg sbt",
        );

        let hg_sbt = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR,
            (hit_group_handles.len() * size_of_val(&hit_group_handles[0])) as vk::DeviceSize,
            debug_utils,
            Some(raytracing_properties.shader_group_base_alignment as u64),
            "hg sbt",
        );

        let tlas = vulkan_objects::TLAS::new(
            device,
            allocator,
            &accel_struct_device,
            &instances,
            scratch_buffer_alignment as vk::DeviceSize,
            compute_helper,
            debug_utils,
            "TLAS",
        );

        // record batch and submit
        data_helper.record_batch(device);
        data_helper.copy_buffer_to_buffer(
            device,
            staging_rg_sbt.vk_buffer(),
            rg_sbt.vk_buffer(),
            staging_rg_sbt.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_ms_sbt.vk_buffer(),
            ms_sbt.vk_buffer(),
            staging_ms_sbt.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_hg_sbt.vk_buffer(),
            hg_sbt.vk_buffer(),
            staging_hg_sbt.data_size,
        );
        data_helper.submit_batch(device, None, None, None);

        allocator.destroy_buffer_resource(device, staging_rg_sbt);
        allocator.destroy_buffer_resource(device, staging_ms_sbt);
        allocator.destroy_buffer_resource(device, staging_hg_sbt);

        // create the strided region to be passed to trace rays
        let rg_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(rg_sbt.device_address)
            .stride(aligned_size!(
                raytracing_properties.shader_group_handle_size,
                raytracing_properties.shader_group_handle_alignment
            ) as vk::DeviceSize)
            .size(aligned_size!(
                raytracing_properties.shader_group_handle_size,
                raytracing_properties.shader_group_handle_alignment
            ) as vk::DeviceSize);

        let ms_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(ms_sbt.device_address)
            .stride(aligned_size!(
                raytracing_properties.shader_group_handle_size,
                raytracing_properties.shader_group_handle_alignment
            ) as vk::DeviceSize)
            .size(ms_group_handles.len() as vk::DeviceSize);

        let hg_region = vk::StridedDeviceAddressRegionKHR::default()
            .device_address(hg_sbt.device_address)
            .stride(aligned_size!(
                raytracing_properties.shader_group_handle_size + size_of::<HGRecordData>() as u32,
                raytracing_properties.shader_group_handle_alignment
            ) as vk::DeviceSize)
            .size(hit_group_handles.len() as vk::DeviceSize);

        // create descriptor sets
        let (descriptor_pool, descriptor_set) =
            Scene::create_descriptor_sets(&device, &common_scene_data, &pipeline_data, &tlas);

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(descriptor_pool, "render scene descriptor pool");
            debug_utils.set_object_name(descriptor_set, "render scene descriptor set");
        }

        let textures_count = common_scene_data.images_descriptor_info.len() as u32;

        Self {
            pipeline_data,
            descriptor_pool,
            descriptor_set,
            rg_sbt: *rg_sbt.vk_buffer(),
            ms_sbt: *ms_sbt.vk_buffer(),
            hg_sbt: *hg_sbt.vk_buffer(),
            rg_sbt_allocation: rg_sbt.allocation_type,
            ms_sbt_allocation: ms_sbt.allocation_type,
            hg_sbt_allocation: hg_sbt.allocation_type,
            positions: *positions_resource.vk_buffer(),
            positions_allocation: positions_resource.allocation_type,
            indices: *indices_resource.vk_buffer(),
            indices_allocation: indices_resource.allocation_type,
            vertices_data: *vertices_data_resource.vk_buffer(),
            vertices_data_allocation: vertices_data_resource.allocation_type,
            light_bboxes: *light_bboxes_resource.vk_buffer(),
            light_bboxes_allocation: light_bboxes_resource.allocation_type,
            blases,
            tlas,
            accel_struct_device,
            raytracing_device,
            rg_region,
            ms_region,
            hg_region,
            textures_count,
            common_scene_data: common_scene_data.clone(),
        }
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        common_scene_data: &viewport::CommonSceneData,
        pipeline_data: &PipelineData,
        tlas: &vulkan_objects::TLAS,
    ) -> (vk::DescriptorPool, vk::DescriptorSet) {
        let descriptor_pool;
        let descriptor_set;

        unsafe {
            descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize::default()
                                .descriptor_count(2)
                                .ty(vk::DescriptorType::STORAGE_IMAGE),
                            vk::DescriptorPoolSize::default()
                                .descriptor_count(1)
                                .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                            vk::DescriptorPoolSize::default()
                                .descriptor_count(
                                    common_scene_data.images_descriptor_info.len() as u32
                                )
                                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                        ]),
                    None,
                )
                .unwrap();

            let ds_counts = [common_scene_data.images_descriptor_info.len() as u32];
            let mut tex_ds_ai = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&ds_counts);

            descriptor_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .push_next(&mut tex_ds_ai)
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[pipeline_data.descriptor_set_layout]),
                )
                .unwrap()[0];

            let accel_structs = [tlas.accel_struct];
            let mut tlas_desc_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&accel_structs);

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .descriptor_count(1)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .dst_binding(2)
                    .dst_set(descriptor_set)
                    .push_next(&mut tlas_desc_info),
                vk::WriteDescriptorSet::default()
                    .descriptor_count(common_scene_data.images_descriptor_info.len() as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .dst_binding(3)
                    .dst_set(descriptor_set)
                    .image_info(&common_scene_data.images_descriptor_info),
            ];

            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        (descriptor_pool, descriptor_set)
    }

    pub fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        unsafe {
            device.queue_wait_idle(*queue).unwrap();
        }

        self.pipeline_data.destroy(device);
        self.pipeline_data = PipelineData::new(
            device,
            debug_utils,
            self.textures_count,
            &self.raytracing_device,
        );

        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        (self.descriptor_pool, self.descriptor_set) = Scene::create_descriptor_sets(
            device,
            &self.common_scene_data,
            &self.pipeline_data,
            &self.tlas,
        );

        // let the new pipeline know about accumalation target
        self.update_accum_target_desc(device, accum_target);
        // let the new pipeline know about final render target
        self.update_final_render_target_desc(device, final_render_target);
    }

    pub fn update_accum_target_desc(
        &self,
        device: &ash::Device,
        accum_target: vk::DescriptorImageInfo,
    ) {
        let image_infos = [accum_target];
        let descriptor_writes = [vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .dst_set(self.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos)];

        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    pub fn update_final_render_target_desc(
        &self,
        device: &ash::Device,
        final_render_target: vk::DescriptorImageInfo,
    ) {
        let image_infos = [final_render_target];
        let descriptor_writes = [vk::WriteDescriptorSet::default()
            .descriptor_count(1)
            .dst_set(self.descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&image_infos)];

        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }
    }

    pub fn render(
        &self,
        device: &ash::Device,
        random_states: vk::DeviceAddress,
        accum_target: vk::DescriptorImageInfo,
        final_render_target: vk::DescriptorImageInfo,
        render_extent: vk::Extent2D,
        command_buffer: vk::CommandBuffer,
        current_sample: u32,
        render_mode: u32,
        camera_index: u32,
        bounce_count: u32,
        common_scene_data: &viewport::CommonSceneData,
    ) {
        let pc = PushConstants {
            camera_instances: common_scene_data.camera_instances_addr,
            punctual_lights: common_scene_data.punctual_lights_addr,
            mesh_emitters: common_scene_data.mesh_emitters_addr,
            punctual_lights_count: common_scene_data.punctual_lights_count,
            mesh_emitters_count: common_scene_data.mesh_emitters_count,
            materials: common_scene_data.materials_addr,
            random_states,
            current_sample,
            render_mode,
            camera_index,
            bounce_count,
        };

        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_data.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_data.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                command_buffer,
                self.pipeline_data.layout,
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::ANY_HIT_KHR
                    | vk::ShaderStageFlags::INTERSECTION_KHR,
                0,
                bytemuck::bytes_of(&pc),
            );

            self.raytracing_device.cmd_trace_rays(
                command_buffer,
                &self.rg_region,
                &self.ms_region,
                &self.hg_region,
                &vk::StridedDeviceAddressRegionKHR::default(),
                render_extent.width,
                render_extent.height,
                1,
            );
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        self.pipeline_data.destroy(device);

        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        for blas in self.blases.iter_mut() {
            blas.destroy(device, &self.accel_struct_device, allocator);
        }

        self.tlas
            .destroy(device, &self.accel_struct_device, allocator);

        allocator.destroy_buffer_and_allocation(device, self.rg_sbt, &mut self.rg_sbt_allocation);
        allocator.destroy_buffer_and_allocation(device, self.ms_sbt, &mut self.ms_sbt_allocation);
        allocator.destroy_buffer_and_allocation(device, self.hg_sbt, &mut self.hg_sbt_allocation);

        allocator.destroy_buffer_and_allocation(
            device,
            self.positions,
            &mut self.positions_allocation,
        );
        allocator.destroy_buffer_and_allocation(device, self.indices, &mut self.indices_allocation);
        allocator.destroy_buffer_and_allocation(
            device,
            self.vertices_data,
            &mut self.vertices_data_allocation,
        );
        allocator.destroy_buffer_and_allocation(
            device,
            self.light_bboxes,
            &mut self.light_bboxes_allocation,
        );
    }
}
