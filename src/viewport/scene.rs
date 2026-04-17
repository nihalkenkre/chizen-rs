use gltf::json::extensions::mesh;
use nalgebra_glm::{self as glm};
use std::{f64::consts::E, ffi::CStr};

use crate::{
    viewport::{self},
    vulkan_objects::{self, AllocatorTrait},
};
use ash::vk::{self};

#[derive(Default)]
pub enum SceneType {
    World(Scene),
    #[default]
    None,
}

pub trait SceneTrait {
    fn camera_names(&self) -> &[String];
    fn common_scene_data(&self) -> viewport::CommonSceneData;
    fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        debug_utils: &vulkan_objects::DebugUtils,
    );
    fn render(&self, device: &ash::Device, command_buffer: &vk::CommandBuffer, camera_index: usize);
    fn destroy(self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType);
}

/// Storing pre multiplied matrices here to save on doing the multiplications on the GPU
/// and might introduce unwanted complications in the shader code.
/// Prima facie this leads to duplication of data since many camera nodes might reference a single camera.
/// Need to investigate
/// Sent to shader, array of 4 values to satisfy glsl alignments
#[derive(Debug, Default)]
struct CameraInstance {
    pub view_projection_matrix: glm::Mat4,
    pub view_inverse_matrix: glm::Mat4,
    pub projection_inverse_matrix: glm::Mat4,

    // [0]: z_near, [1]: z_far, [2]: free to use, [3]: free to use
    pub z_near_far: [f32; 4],
}

/// Sent to shader, array of 4 values to satisfy glsl alignments
#[derive(Debug, Default)]
struct PunctualLight {
    pub position: [f32; 4],
    pub direction: [f32; 4],
    pub scale: [f32; 4],
    pub power: [f32; 4], // color * intensity
    pub bbox_min: [f32; 4],
    pub bbox_max: [f32; 4],
    // [0]: inner_cone_angle, [1]: outer_cone_angle, [2]: radius
    pub icacos_ocacos_radius: [f32; 4],
    // 0: Directional, 1: Point, 2: Spot, 3: Area
    pub ty: u32,
}

/// Hold a volume to sample the light/shadow rays from, if the emissive factor > 0.
/// Actual direct lighting happens when the ray collides with the geometry and the emissive.
/// Sent to shader, array of 4 values to satisfy glsl alignments
#[derive(Debug, Default)]
struct MeshEmitter {
    pub bounding_box_min: [f32; 4],
    pub bounding_box_max: [f32; 4],
    // [0]: area, [1-2-3]: avg_emission
    pub area_avgemission: [f32; 4],
}

/// Sent to shader, array of 4 values to satisfy glsl alignments
#[derive(Debug, Default)]
struct Material {
    pub base_color_factor: [f32; 4],
    pub emissive_factor: [f32; 4],
    // [0]: base_color_index, [1]: normal_index, [2]: metalrough_index, [3]: emissive_index
    pub base_color_normal_metalrough_emissive_index: [u32; 4],
    // [0]: metal_factor, [1]: rough_factor, [2]: transmission factor, [3]: ior
    pub metal_rough_transmission_ior_factor: [f32; 4],
    // [0]: transmission_index
    pub tranmission_index: [u32; 4],
}

#[derive(Debug, Default)]
struct PipelineData {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

#[derive(Debug, Default, Clone, Copy)]
struct PushConstants {
    pub camera_instances: vk::DeviceAddress,
    pub model_matrices: vk::DeviceAddress,
    pub punctual_lights: vk::DeviceAddress,
    pub mesh_emitters: vk::DeviceAddress,
    pub materials: vk::DeviceAddress,

    pub positions: vk::DeviceAddress,
    pub meshlets: vk::DeviceAddress,
    pub meshlets_triangles: vk::DeviceAddress,
    pub meshlets_vertices: vk::DeviceAddress,
    pub vertices_data: vk::DeviceAddress,

    pub camera_index: u32,
    pub punctual_lights_count: u32,
    pub mesh_emitters_count: u32,
    pub meshlets_count: u32,
}

impl SceneTrait for SceneType {
    fn camera_names(&self) -> &[String] {
        match self {
            SceneType::World(scene) => &scene.camera_names,
            SceneType::None => &[],
        }
    }

    fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        match self {
            SceneType::World(scene) => {
                scene.reload_shaders(device, queue, debug_utils);
            }
            SceneType::None => {}
        }
    }

    fn common_scene_data(&self) -> viewport::CommonSceneData {
        match self {
            SceneType::World(scene) => scene.common_scene_data(),
            SceneType::None => viewport::CommonSceneData {
                camera_instances_addr: 0,
                punctual_lights_addr: 0,
                mesh_emitters_addr: 0,
                materials_addr: 0,
                model_matrices_addr: 0,
                images_descriptor_info: vec![],
                punctual_lights_count: 0,
                mesh_emitters_count: 0,
            },
        }
    }

    fn render(
        &self,
        device: &ash::Device,
        command_buffer: &vk::CommandBuffer,
        camera_index: usize,
    ) {
        match self {
            SceneType::World(scene) => scene.render(device, command_buffer, camera_index),
            SceneType::None => {}
        }
    }

    fn destroy(self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        match self {
            SceneType::World(scene) => scene.destroy(device, allocator),
            SceneType::None => {}
        }
    }
}

unsafe impl bytemuck::NoUninit for PushConstants {}

impl PipelineData {
    pub fn new(
        device: &ash::Device,
        debug_utils: &vulkan_objects::DebugUtils,
        textures_count: u32,
    ) -> Self {
        let spirv_data = std::fs::read(
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
                + ("/shaders/viewport.slang.spv"),
        )
        .expect("Could not read viewport.slang.spv");
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
                .stage(vk::ShaderStageFlags::TASK_EXT)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("task_main\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MESH_EXT)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("mesh_main\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("fragment_main\0".as_bytes()).unwrap()),
        ];
        let ras_ci = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1f32)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let ms_ci = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let dss = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .max_depth_bounds(1f32);
        let cbas = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )];

        let cbs_ci = vk::PipelineColorBlendStateCreateInfo::default().attachments(&cbas);
        let ds = [
            vk::DynamicState::VIEWPORT_WITH_COUNT,
            vk::DynamicState::SCISSOR_WITH_COUNT,
        ];

        let ds_ci = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&ds);

        let dsl_0_binds = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .descriptor_count(textures_count)];

        let binding_flags = [vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND];

        let mut dsl_0_binds_flags_ci =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let dsl_cis = [vk::DescriptorSetLayoutCreateInfo::default()
            .push_next(&mut dsl_0_binds_flags_ci)
            .bindings(&dsl_0_binds)];

        let mut descriptor_set_layouts = vec![];

        for dsl_ci in dsl_cis {
            unsafe {
                descriptor_set_layouts
                    .push(device.create_descriptor_set_layout(&dsl_ci, None).unwrap());
            }
        }
        let pc_ranges = [vk::PushConstantRange::default()
            .stage_flags(
                vk::ShaderStageFlags::TASK_EXT
                    | vk::ShaderStageFlags::MESH_EXT
                    | vk::ShaderStageFlags::FRAGMENT,
            )
            .size(size_of::<PushConstants>() as u32)];

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&pc_ranges)
            .set_layouts(&descriptor_set_layouts);

        let layout;
        unsafe {
            layout = device.create_pipeline_layout(&layout_ci, None).unwrap();
        }

        let col_attach_formats = [vk::Format::R8G8B8A8_SRGB];

        let mut rend_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&col_attach_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let cis = [vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut rend_info)
            .stages(&stages)
            .rasterization_state(&ras_ci)
            .multisample_state(&ms_ci)
            .depth_stencil_state(&dss)
            .color_blend_state(&cbs_ci)
            .dynamic_state(&ds_ci)
            .layout(layout)];

        let pipeline;
        unsafe {
            pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &cis, None)
                .unwrap()[0];

            device.destroy_shader_module(shader_mod, None);
        }

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(pipeline, "viewport scene pipeline");
            debug_utils.set_object_name(layout, "viewport scene pipeline layout");

            for dsl in &descriptor_set_layouts {
                debug_utils.set_object_name(*dsl, "viewport scene dsl");
            }
        }

        Self {
            pipeline,
            layout,
            descriptor_set_layouts,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            for dsl in &self.descriptor_set_layouts {
                device.destroy_descriptor_set_layout(*dsl, None);
            }

            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct Scene {
    pub camera_instances: vulkan_objects::BufferResource,
    pub model_matrices: vulkan_objects::BufferResource,
    pub materials_buffer: vulkan_objects::BufferResource,
    pub meshlets_positions_buffer: vulkan_objects::BufferResource,
    pub meshlets_vertices_data_buffer: vulkan_objects::BufferResource,
    pub meshlets_buffer: vulkan_objects::BufferResource,
    pub meshlets_vertices_buffer: vulkan_objects::BufferResource,
    pub meshlets_triangles_buffer: vulkan_objects::BufferResource,
    pub meshlets_count: u32,
    pub punctual_lights: vulkan_objects::BufferResource,
    pub mesh_emitters: vulkan_objects::BufferResource,
    pub punctual_lights_count: u32,
    pub mesh_emitters_count: u32,

    pub meshlets: meshopt::Meshlets,

    pub camera_names: Vec<String>,

    pipeline_data: PipelineData,
    mesh_shader_device: Option<ash::ext::mesh_shader::Device>,

    image_resources: Vec<vulkan_objects::ImageResource>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
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
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
        compute_queue: &vk::Queue,
        compute_queue_family_index: u32,
    ) -> Self {
        let mut materials = vec![];

        let mut staging_image_buffers = vec![];
        let mut image_resources: Vec<vulkan_objects::ImageResource> = vec![];
        let mut images_data: Vec<(gltf::image::Format, u32, u32, bool)> = vec![]; // (format, width, height, should do srgb_to_linear)

        // default white texture, used when no texture is assigned in the scene.
        {
            let data: [u8; 4] = [255, 255, 255, 255];
            staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &data,
                debug_utils,
                None,
                "default white texture staging buffer",
            ));

            let ptr = staging_image_buffers[0].mapped_ptr.unwrap() as *mut u8;

            image_resources.push(allocator.create_image(
                device,
                &vk::Extent2D::default().width(1).height(1),
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED,
                queue_family_indices,
                debug_utils,
                "default white texture",
            ));
        }
        images_data.push((gltf::image::Format::R8G8B8A8, 1, 1, false));

        // default red material.
        // all primitives are assigned this material on initialization and later updated to the scene material.
        materials.push(Material {
            base_color_factor: [1f32, 0f32, 0f32, 1f32],
            metal_rough_transmission_ior_factor: [1f32, 1f32, 0f32, 1.5],
            ..Default::default()
        });

        // import the materials and relevant textures.
        // The textures are stored as entries for the data converter to run through them.
        for material in gltf.materials() {
            let pbr_metal_rough = material.pbr_metallic_roughness();

            let base_color_index = match pbr_metal_rough.base_color_texture() {
                Some(texture) => {
                    let curr_image = images.get(texture.texture().source().index()).unwrap();

                    staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                        device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &curr_image.pixels,
                        debug_utils,
                        None,
                        "staging image",
                    ));
                    // Using UNORM, because we're performing srgb to linear in the data_conversion shader.
                    // Using SRGB here, prevents _STORAGE which is required for data conversion(3 channel to 4 channel),
                    // which in turn is required to convert all kinds of channels to 4 channels.
                    // Another option is to create the RGBA image on the CPU, create a SRGB, and use buffer to image copy.
                    // hmmm :?, might be horrendously slow... might be ???
                    image_resources.push(
                        allocator.create_image(
                            device,
                            &vk::Extent2D::default()
                                .width(curr_image.width)
                                .height(curr_image.height),
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                            queue_family_indices,
                            debug_utils,
                            texture.texture().source().name().unwrap_or("default"),
                        ),
                    );

                    images_data.push((
                        curr_image.format,
                        curr_image.width,
                        curr_image.height,
                        true,
                    ));

                    (image_resources.len() - 1) as u32
                }
                None => 0,
            };

            let normal_index = match material.normal_texture() {
                Some(texture) => {
                    let curr_image = images.get(texture.texture().source().index()).unwrap();

                    staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                        device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &curr_image.pixels,
                        debug_utils,
                        None,
                        "staging image",
                    ));

                    image_resources.push(
                        allocator.create_image(
                            device,
                            &vk::Extent2D::default()
                                .width(curr_image.width)
                                .height(curr_image.height),
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                            queue_family_indices,
                            debug_utils,
                            texture.texture().source().name().unwrap_or("default"),
                        ),
                    );

                    images_data.push((
                        curr_image.format,
                        curr_image.width,
                        curr_image.height,
                        false,
                    ));

                    (image_resources.len() - 1) as u32
                }
                None => 0,
            };

            let metalrough_index = match pbr_metal_rough.metallic_roughness_texture() {
                Some(texture) => {
                    let curr_image = images.get(texture.texture().source().index()).unwrap();

                    staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                        device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &curr_image.pixels,
                        debug_utils,
                        None,
                        "staging image",
                    ));

                    image_resources.push(
                        allocator.create_image(
                            device,
                            &vk::Extent2D::default()
                                .width(curr_image.width)
                                .height(curr_image.height),
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                            queue_family_indices,
                            debug_utils,
                            texture.texture().source().name().unwrap_or("default"),
                        ),
                    );

                    images_data.push((
                        curr_image.format,
                        curr_image.width,
                        curr_image.height,
                        false,
                    ));

                    (image_resources.len() - 1) as u32
                }
                None => 0,
            };

            let emissive_index = match material.emissive_texture() {
                Some(texture) => {
                    let curr_image = images.get(texture.texture().source().index()).unwrap();

                    staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                        device,
                        vk::BufferUsageFlags::TRANSFER_SRC,
                        &curr_image.pixels,
                        debug_utils,
                        None,
                        "staging image",
                    ));

                    image_resources.push(
                        allocator.create_image(
                            device,
                            &vk::Extent2D::default()
                                .width(curr_image.width)
                                .height(curr_image.height),
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::TRANSFER_DST
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED,
                            queue_family_indices,
                            debug_utils,
                            texture.texture().source().name().unwrap_or("default"),
                        ),
                    );

                    images_data.push((
                        curr_image.format,
                        curr_image.width,
                        curr_image.height,
                        true,
                    ));

                    (image_resources.len() - 1) as u32
                }
                None => 0,
            };

            let (transmission_index, transmission_factor) = match material.transmission() {
                Some(transmission) => match transmission.transmission_texture() {
                    Some(texture) => {
                        let curr_image = images.get(texture.texture().source().index()).unwrap();

                        staging_image_buffers.push(allocator.create_buffer_on_host_with_data(
                            device,
                            vk::BufferUsageFlags::TRANSFER_SRC,
                            &curr_image.pixels,
                            debug_utils,
                            None,
                            "staging image",
                        ));

                        image_resources.push(
                            allocator.create_image(
                                device,
                                &vk::Extent2D::default()
                                    .width(curr_image.width)
                                    .height(curr_image.height),
                                vk::Format::R8G8B8A8_UNORM,
                                vk::ImageUsageFlags::TRANSFER_DST
                                    | vk::ImageUsageFlags::STORAGE
                                    | vk::ImageUsageFlags::SAMPLED,
                                queue_family_indices,
                                debug_utils,
                                texture.texture().source().name().unwrap_or("default"),
                            ),
                        );

                        images_data.push((
                            curr_image.format,
                            curr_image.width,
                            curr_image.height,
                            true,
                        ));

                        (
                            (image_resources.len() - 1) as u32,
                            transmission.transmission_factor(),
                        )
                    }
                    None => (0, transmission.transmission_factor()),
                },
                None => (0, 0f32),
            };

            let ior = match material.ior() {
                Some(ior) => ior,
                None => 1.5f32,
            };

            let curr_mat = Material {
                base_color_factor: pbr_metal_rough.base_color_factor(),
                emissive_factor: [
                    material.emissive_factor()[0] * material.emissive_strength().unwrap_or(1f32),
                    material.emissive_factor()[1] * material.emissive_strength().unwrap_or(1f32),
                    material.emissive_factor()[2] * material.emissive_strength().unwrap_or(1f32),
                    1f32,
                ],
                base_color_normal_metalrough_emissive_index: [
                    base_color_index,
                    normal_index,
                    metalrough_index,
                    emissive_index,
                ],
                metal_rough_transmission_ior_factor: [
                    pbr_metal_rough.metallic_factor(),
                    pbr_metal_rough.roughness_factor(),
                    transmission_factor,
                    ior,
                ],
                tranmission_index: [transmission_index, 0, 0, 0],
            };

            materials.push(curr_mat);
        }

        // gather all the image descriptors, these would be used to update the descriptor set
        let mut image_descs = vec![];
        for image_resource in &image_resources {
            image_descs.push(image_resource.descriptor_info);
        }

        // initialize a data converter object to convert the batched images
        let mut data_converter = vulkan_objects::DataConverter::new(
            device,
            compute_queue,
            compute_queue_family_index,
            &debug_utils,
            &image_descs,
        );

        data_converter.record_batch(device);

        for r in 0..image_resources.len() as usize {
            match images_data.get(r).unwrap().0 {
                gltf::image::Format::R8
                | gltf::image::Format::R8G8
                | gltf::image::Format::R8G8B8
                | gltf::image::Format::R8G8B8A8 => {
                    data_converter.change_image_layout(
                        device,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                        vk::AccessFlags2::SHADER_WRITE,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::GENERAL,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::QUEUE_FAMILY_IGNORED,
                        vk::ImageAspectFlags::COLOR,
                        &image_resources.get(r).unwrap().image,
                    );

                    data_converter.convert(
                        device,
                        staging_image_buffers.get(r).unwrap().device_address,
                        r as u32,
                        *images_data.get(r).unwrap(),
                    );
                }
                _ => {}
            }
        }

        // all images are linear rgba device vulkan images
        data_converter.submit_batch(device, None, None, None);

        for staging_image_buffer in staging_image_buffers {
            allocator.destroy_buffer_resource(device, staging_image_buffer);
        }

        // create pipeline for the viewport
        let pipeline_data = PipelineData::new(device, debug_utils, image_resources.len() as u32);

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .descriptor_count(image_resources.len() as u32)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];

        let dsp_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool;
        let descriptor_set;

        unsafe {
            descriptor_pool = device.create_descriptor_pool(&dsp_ci, None).unwrap();
            let ds_counts = [image_resources.len() as u32];
            let mut tex_ds_ai = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&ds_counts);

            let ds_ai = vk::DescriptorSetAllocateInfo::default()
                .push_next(&mut tex_ds_ai)
                .descriptor_pool(descriptor_pool)
                .set_layouts(&pipeline_data.descriptor_set_layouts);
            descriptor_set = device.allocate_descriptor_sets(&ds_ai).unwrap()[0];

            if cfg!(debug_assertions) {
                debug_utils
                    .set_object_name(descriptor_pool, "viewport merged scene descriptor pool");
                debug_utils.set_object_name(descriptor_set, "viewport merged scene descriptor set");
            }
        }

        // to update descriptor set
        let mut image_infos = Vec::with_capacity(image_resources.len());

        for r in 0..image_resources.len() as usize {
            image_infos.push(image_resources.get(r).unwrap().descriptor_info)
        }

        let descriptor_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .descriptor_count(image_resources.len() as u32)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        unsafe {
            device.update_descriptor_sets(&[descriptor_write], &[]);
        }

        // look to get other scene data, run through the nodes in the gltf file and take what is required
        // meshes, camera_instances, model_matrices, punctual_lights, mesh_emitters (common_scene_data) are passed to the viewport shader and the raytracing shader
        let mut camera_instances = vec![];
        let mut model_matrices = vec![];
        let mut punctual_lights = vec![];
        let mut mesh_emitters = vec![];
        let mut positions: Vec<[f32; 3]> = vec![];
        let mut indices = vec![];
        let mut vertices_data = vec![];
        let mut camera_names = vec![];
        let mut vertex_count = 0;

        for node in gltf.nodes() {
            // Camera
            if let Some(camera) = node.camera() {
                let mut z_near_far = [0.0, 0.0, 0.0, 0.0];
                let z_near;
                let z_far;

                let view_matrix =
                    glm::inverse(&glm::make_mat4(node.transform().matrix().as_flattened()));
                let mut projection_matrix;

                match camera.projection() {
                    gltf::camera::Projection::Orthographic(ortho) => {
                        projection_matrix = glm::ortho(
                            -ortho.xmag() / 2f32,
                            ortho.xmag() / 2f32,
                            -ortho.ymag() / 2f32,
                            -ortho.ymag() / 2f32,
                            ortho.znear(),
                            ortho.zfar(),
                        );

                        z_near_far[0] = ortho.znear();
                        z_near_far[1] = ortho.zfar();

                        z_near = ortho.znear();
                        z_far = ortho.zfar();
                    }
                    gltf::camera::Projection::Perspective(persp) => {
                        projection_matrix = glm::perspective(
                            persp.aspect_ratio().unwrap_or(16f32 / 9f32),
                            persp.yfov(),
                            persp.znear(),
                            persp.zfar().unwrap_or(1000000f32),
                        );

                        z_near_far[0] = persp.znear();
                        z_near_far[1] = persp.zfar().unwrap_or(1000000f32);

                        z_near = persp.znear();
                        z_far = persp.zfar().unwrap_or(1000000f32);
                    }
                };

                projection_matrix[1 * 4 + 1] *= -1.0;

                let view_projection_matrix = projection_matrix * view_matrix;
                let view_inverse_matrix = glm::inverse(&view_matrix); // used in the raytracing shader
                let projection_inverse_matrix = glm::inverse(&projection_matrix); // used in the raytracing shader
                let name = node.name().unwrap_or("default camera").to_owned();

                camera_instances.push(CameraInstance {
                    view_projection_matrix,
                    view_inverse_matrix,
                    projection_inverse_matrix,
                    z_near_far: [z_near, z_far, 0f32, 0f32],
                });

                camera_names.push(name);
            }
            //Mesh
            else if let Some(curr_mesh) = node.mesh() {
                model_matrices.push(glm::make_mat4(node.transform().matrix().as_flattened()));

                for curr_prim in curr_mesh.primitives() {
                    let reader = curr_prim.reader(|buffer| Some(&buffers[buffer.index()]));
                    let mut curr_pos = reader.read_positions().unwrap().collect::<Vec<[f32; 3]>>();

                    let mut curr_tan = vec![];
                    // if tangents not found create an array with zeros, required for VertexData struct
                    if let Some(t) = reader.read_tangents() {
                        curr_tan = t.collect::<Vec<[f32; 4]>>();
                    } else {
                        eprintln!(
                            "No tangents for {:?}",
                            curr_mesh.name().unwrap_or("default mesh")
                        );

                        curr_tan = vec![[0f32, 0f32, 0f32, 0f32]; curr_pos.len()];
                    }

                    let curr_norm = reader.read_normals().unwrap().collect::<Vec<[f32; 3]>>();

                    let mut curr_uvs_0 = vec![];
                    // if uvs not found create an array with zeros, required for VertexData struct
                    if let Some(u) = reader.read_tex_coords(0) {
                        curr_uvs_0 = u.into_f32().collect::<Vec<[f32; 2]>>()
                    } else {
                        println!(
                            "No uvs_0 for {:?}",
                            curr_mesh.name().unwrap_or("default mesh")
                        );

                        curr_uvs_0 = vec![[0f32, 0f32]; curr_pos.len()];
                    }

                    let material_index = match curr_prim.material().index() {
                        Some(index) => (index + 1) as u32, // + 1 since index 0 is the default material
                        None => 0,
                    };

                    for v in 0..curr_pos.len() {
                        vertices_data.push(viewport::VertexData {
                            tangent: curr_tan[v],
                            normal: [curr_norm[v][0], curr_norm[v][1], curr_norm[v][2], 0f32],
                            uv_0: curr_uvs_0[v],
                            model_matrix_material_index: [
                                (model_matrices.len() - 1) as u32,
                                material_index,
                            ],
                        });
                    }

                    // offset the index values as per vertex count.
                    let mut curr_indices = reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .collect::<Vec<u32>>()
                        .iter()
                        .map(|index| *index + vertex_count as u32)
                        .collect::<Vec<u32>>();

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

                    // calculate area of the mesh, and add it to mesh emitter list.
                    if avg_emission[0] > 0f32 || avg_emission[1] > 0f32 || avg_emission[2] > 0f32 {
                        let mut tri_areas = vec![];

                        let position = node.transform().decomposed().0;
                        let scale = node.transform().decomposed().2;
                        let scale_mat = glm::scale(&glm::identity(), &glm::make_vec3(&scale));

                        let tri_count = curr_indices.len() / 3;
                        let mut prim_area = 0f32;

                        for t in 0..tri_count {
                            let v1_tmp = glm::make_vec3(
                                curr_pos
                                    .get(*curr_indices.get(t * 3).unwrap() as usize - vertex_count)
                                    .unwrap(),
                            );
                            let v1 = scale_mat * glm::vec4(v1_tmp.x, v1_tmp.y, v1_tmp.z, 0f32);

                            let v2_tmp = glm::make_vec3(
                                curr_pos
                                    .get(
                                        *curr_indices.get(t * 3 + 1).unwrap() as usize
                                            - vertex_count,
                                    )
                                    .unwrap(),
                            );
                            let v2 = scale_mat * glm::vec4(v2_tmp.x, v2_tmp.y, v2_tmp.z, 0f32);

                            let v3_tmp = glm::make_vec3(
                                curr_pos
                                    .get(
                                        *curr_indices.get(t * 3 + 2).unwrap() as usize
                                            - vertex_count,
                                    )
                                    .unwrap(),
                            );
                            let v3 = scale_mat * glm::vec4(v3_tmp.x, v3_tmp.y, v3_tmp.z, 0f32);

                            let v2v1 = v2 - v1;
                            let v3v1 = v3 - v1;

                            let tri_area = glm::magnitude(&glm::cross(
                                &glm::vec3(v3v1.x, v3v1.y, v3v1.z),
                                &glm::vec3(v2v1.x, v2v1.y, v2v1.z),
                            )) * 0.5;

                            prim_area += tri_area;
                            tri_areas.push(tri_area);
                        }

                        let mut tri_probs = vec![];

                        for tri_area in &tri_areas {
                            tri_probs.push(tri_area / prim_area);
                        }

                        let mut tri_cdf = vec![];

                        let mut cdf = 0f32;
                        for tri_prob in &tri_probs {
                            cdf += tri_prob;
                            tri_cdf.push(cdf);
                        }

                        mesh_emitters.push(MeshEmitter {
                            bounding_box_min: [
                                position[0] + (curr_prim.bounding_box().min[0] * scale[0]),
                                position[1] + (curr_prim.bounding_box().min[1] * scale[1]),
                                position[2] + (curr_prim.bounding_box().min[2] * scale[2]),
                                0f32,
                            ],
                            bounding_box_max: [
                                position[0] + (curr_prim.bounding_box().max[0] * scale[0]),
                                position[1] + (curr_prim.bounding_box().max[1] * scale[1]),
                                position[2] + (curr_prim.bounding_box().max[2] * scale[2]),
                                0f32,
                            ],
                            area_avgemission: [
                                prim_area,
                                avg_emission[0],
                                avg_emission[1],
                                avg_emission[2],
                            ],
                        });
                    }

                    vertex_count += curr_pos.len();

                    positions.append(&mut curr_pos);
                    indices.append(&mut curr_indices);
                }
            }
            // Punctual Light
            else if let Some(light) = node.light() {
                let (ty, inner_cone_angle_cos, outer_cone_angle_cos) = match light.kind() {
                    gltf::khr_lights_punctual::Kind::Directional => (0, 0f32, 0f32),
                    gltf::khr_lights_punctual::Kind::Point => (1, 0f32, 0f32),
                    gltf::khr_lights_punctual::Kind::Spot {
                        inner_cone_angle,
                        outer_cone_angle,
                    } => (2, inner_cone_angle.cos(), outer_cone_angle.cos()),
                };

                let color = [light.color()[0], light.color()[1], light.color()[2], 1f32];
                let intensity = light.intensity();
                let radius = match light.extras() {
                    Some(extras) => {
                        let v: serde_json::Value = serde_json::from_str(extras.get()).unwrap();

                        match v {
                            serde_json::Value::Object(v) => {
                                v.get("radius")
                                    .unwrap_or(&serde_json::to_value(0.01f32).unwrap())
                                    .as_f64()
                                    .unwrap()
                                    .max(0.01) as f32
                            }
                            _ => 0.01,
                        }
                    }
                    None => 0.01,
                };

                let position = [
                    node.transform().decomposed().0[0],
                    node.transform().decomposed().0[1],
                    node.transform().decomposed().0[2],
                    1f32,
                ];

                let quat = node.transform().decomposed().1;
                let scale = [
                    node.transform().decomposed().2[0],
                    node.transform().decomposed().2[1],
                    node.transform().decomposed().2[2],
                    1f32,
                ];

                let direction: [f32; 4] = (glm::quat_to_mat4(&glm::make_quat(&quat))
                    * glm::vec4(0f32, 0f32, -1f32, 1f32))
                .into();

                let power = match ty {
                    0 => [
                        color[0] * intensity / 683f32,
                        color[1] * intensity / 683f32,
                        color[2] * intensity / 683f32,
                        color[3] * intensity / 683f32,
                    ],
                    _ => [
                        color[0] * intensity / 54.351414 / 25f32,
                        color[1] * intensity / 54.351414 / 25f32,
                        color[2] * intensity / 54.351414 / 25f32,
                        color[3] * intensity / 54.351414 / 25f32,
                    ],
                };

                punctual_lights.push(PunctualLight {
                    ty,
                    position,
                    direction,
                    scale,
                    power,
                    bbox_min: [
                        position[0] - radius,
                        position[1] - radius,
                        position[2] - radius,
                        0f32,
                    ],
                    bbox_max: [
                        position[0] + radius,
                        position[1] + radius,
                        position[2] + radius,
                        0f32,
                    ],
                    icacos_ocacos_radius: [
                        inner_cone_angle_cos,
                        outer_cone_angle_cos,
                        radius,
                        0f32,
                    ],
                });
            }
        }

        // Default camera, when there is no camera in the scene.
        // eye - 10 10 10, center - 0 0 0, fov - 50 aspect ratio 16 / 9
        if camera_instances.len() == 0 {
            let z_near_far = [0.01f32, 1000000f32, 0f32, 0f32];
            let z_near = 0.01f32;
            let z_far = 1000000f32;

            let view_matrix = glm::look_at(
                &glm::vec3(10f32, 10f32, 10f32),
                &glm::vec3(0f32, 0f32, 0f32),
                &glm::vec3(0f32, 1f32, 0f32),
            );
            let view_inverse_matrix = glm::inverse(&view_matrix);
            let projection_matrix = glm::perspective(16f32 / 9f32, 50f32, z_near, z_far);
            let projection_inverse_matrix = glm::inverse(&projection_matrix);

            let view_projection_matrix = projection_matrix * view_matrix;

            camera_instances.push(CameraInstance {
                view_projection_matrix,
                view_inverse_matrix,
                projection_inverse_matrix,
                z_near_far: [z_near, z_far, 0f32, 0f32],
            });
        }

        // prepare to upload the common_scene_data to the device
        let staging_camera_instances = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &camera_instances,
            debug_utils,
            None,
            "staging camera matrices",
        );

        let camera_instances = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (camera_instances.len() * size_of_val(&camera_instances[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "camera matrices",
        );

        let staging_model_matrices = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &model_matrices,
            debug_utils,
            None,
            "staging model matrices",
        );

        let model_matrices = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (model_matrices.len() * size_of_val(&model_matrices[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "model matrices",
        );

        let staging_materials_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &materials,
            debug_utils,
            None,
            "staging materials",
        );

        let materials_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (materials.len() * size_of_val(&materials[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "materials",
        );

        let mut staging_punctual_lights_buffer = vulkan_objects::BufferResource::default();
        let mut punctual_lights_buffer = vulkan_objects::BufferResource::default();

        if punctual_lights.len() > 0 {
            staging_punctual_lights_buffer = allocator.create_buffer_on_host_with_data(
                device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &punctual_lights,
                debug_utils,
                None,
                "staging punctual lights",
            );

            punctual_lights_buffer = allocator.create_buffer_on_device(
                device,
                vk::BufferUsageFlags::TRANSFER_DST,
                (punctual_lights.len() * size_of_val(&punctual_lights[0])) as vk::DeviceSize,
                debug_utils,
                None,
                "punctual lights",
            );
        }

        let mut staging_mesh_emitters_buffer = vulkan_objects::BufferResource::default();
        let mut mesh_emitters_buffer = vulkan_objects::BufferResource::default();

        if mesh_emitters.len() > 0 {
            staging_mesh_emitters_buffer = allocator.create_buffer_on_host_with_data(
                device,
                vk::BufferUsageFlags::TRANSFER_SRC,
                &mesh_emitters,
                debug_utils,
                None,
                "staging mesh emitters",
            );

            mesh_emitters_buffer = allocator.create_buffer_on_device(
                device,
                vk::BufferUsageFlags::TRANSFER_DST,
                (mesh_emitters.len() * size_of_val(&mesh_emitters[0])) as vk::DeviceSize,
                debug_utils,
                None,
                "mesh emitters",
            );
        }

        // create meshlets from the merged scene geometry
        let mut meshlets = meshopt::build_meshlets(
            &indices,
            &meshopt::VertexDataAdapter::new(
                unsafe { &positions.as_flattened().align_to::<u8>().1 },
                size_of_val(&positions[0]),
                0,
            )
            .unwrap(),
            64,
            124,
            0.25,
        );

        let mut meshlets_positions = vec![];
        let mut meshlets_vertices_data = vec![];

        for meshlet in &mut meshlets.meshlets {
            let new_vertex_offset = meshlets_positions.len() as u32;

            for v in 0..meshlet.vertex_count {
                let index = *meshlets
                    .vertices
                    .get((meshlet.vertex_offset + v) as usize)
                    .unwrap() as usize;
                meshlets_positions.push(*positions.get(index).unwrap());
                meshlets_vertices_data.push(*vertices_data.get(index).unwrap());
            }

            meshlet.vertex_offset = new_vertex_offset;
        }

        // pack 3 8 bit indices of a triangle into 1 uint32
        let mut meshlets_triangles = vec![];
        for meshlet in &mut meshlets.meshlets {
            let new_offset = meshlets_triangles.len() as u32;

            for t in 0..meshlet.triangle_count {
                let i0 = t * 3 + meshlet.triangle_offset;
                let i1 = t * 3 + meshlet.triangle_offset + 1;
                let i2 = t * 3 + meshlet.triangle_offset + 2;

                let v_idx_0 = *meshlets.triangles.get(i0 as usize).unwrap() as u32;
                let v_idx_1 = *meshlets.triangles.get(i1 as usize).unwrap() as u32;
                let v_idx_2 = *meshlets.triangles.get(i2 as usize).unwrap() as u32;

                let packed = (v_idx_0 as u32 & 0xFF)
                    | (v_idx_1 as u32 & 0xFF) << 8
                    | (v_idx_2 as u32 & 0xFF) << 16;

                meshlets_triangles.push(packed);
            }

            meshlet.triangle_offset = new_offset;
        }

        // prepare to upload meshlet data to the device
        let staging_meshlets_positions_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &meshlets_positions.as_flattened(),
            debug_utils,
            None,
            "staging meshlets_positions",
        );

        let meshlets_positions_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (meshlets_positions.len() * size_of_val(&meshlets_positions[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "meshlets positions",
        );

        let staging_meshlets_vertices_data_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &meshlets_vertices_data,
            debug_utils,
            None,
            "staging meshlets vertices data",
        );

        let meshlets_vertices_data_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (meshlets_vertices_data.len() * size_of_val(&meshlets_vertices_data[0]))
                as vk::DeviceSize,
            debug_utils,
            None,
            "meshlets vertices data",
        );

        let staging_meshlets_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &meshlets.meshlets,
            debug_utils,
            None,
            "staging meshlets",
        );

        let meshlets_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (meshlets.len() * size_of_val(&meshlets.get(0))) as vk::DeviceSize,
            debug_utils,
            None,
            "meshlets",
        );

        let staging_meshlets_triangles_bufer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &meshlets_triangles,
            debug_utils,
            None,
            "staging meshlets triangles",
        );

        let meshlets_triangles_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (meshlets_triangles.len() * size_of_val(&meshlets_triangles[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "meshlets triangles",
        );

        let staging_meshlets_vertices_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &meshlets.vertices,
            debug_utils,
            None,
            "staging meshlets vertices",
        );

        let meshlets_vertices_buffer = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST,
            (meshlets.vertices.len() * size_of_val(&meshlets.vertices[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "meshlets vertices",
        );

        // record batch and submit
        data_helper.record_batch(device);
        data_helper.copy_buffer_to_buffer(
            device,
            staging_camera_instances.vk_buffer(),
            camera_instances.vk_buffer(),
            staging_camera_instances.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_model_matrices.vk_buffer(),
            model_matrices.vk_buffer(),
            staging_model_matrices.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_materials_buffer.vk_buffer(),
            materials_buffer.vk_buffer(),
            materials_buffer.data_size,
        );
        if punctual_lights.len() > 0 {
            data_helper.copy_buffer_to_buffer(
                device,
                staging_punctual_lights_buffer.vk_buffer(),
                punctual_lights_buffer.vk_buffer(),
                staging_punctual_lights_buffer.data_size,
            );
        }
        if mesh_emitters.len() > 0 {
            data_helper.copy_buffer_to_buffer(
                device,
                staging_mesh_emitters_buffer.vk_buffer(),
                mesh_emitters_buffer.vk_buffer(),
                staging_mesh_emitters_buffer.data_size,
            );
        }
        data_helper.copy_buffer_to_buffer(
            device,
            staging_meshlets_positions_buffer.vk_buffer(),
            meshlets_positions_buffer.vk_buffer(),
            staging_meshlets_positions_buffer.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_meshlets_vertices_data_buffer.vk_buffer(),
            meshlets_vertices_data_buffer.vk_buffer(),
            staging_meshlets_vertices_data_buffer.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_meshlets_buffer.vk_buffer(),
            meshlets_buffer.vk_buffer(),
            staging_meshlets_buffer.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_meshlets_triangles_bufer.vk_buffer(),
            meshlets_triangles_buffer.vk_buffer(),
            staging_meshlets_triangles_bufer.data_size,
        );
        data_helper.copy_buffer_to_buffer(
            device,
            staging_meshlets_vertices_buffer.vk_buffer(),
            meshlets_vertices_buffer.vk_buffer(),
            staging_meshlets_vertices_buffer.data_size,
        );
        data_helper.submit_batch(
            device,
            Some(data_converter.semaphore),
            Some(data_converter.semaphore_value),
            Some(vk::PipelineStageFlags2::TOP_OF_PIPE),
        );

        data_converter.destroy(device);
        allocator.destroy_buffer_resource(device, staging_camera_instances);
        allocator.destroy_buffer_resource(device, staging_model_matrices);
        allocator.destroy_buffer_resource(device, staging_punctual_lights_buffer);
        allocator.destroy_buffer_resource(device, staging_mesh_emitters_buffer);
        allocator.destroy_buffer_resource(device, staging_materials_buffer);
        allocator.destroy_buffer_resource(device, staging_meshlets_positions_buffer);
        allocator.destroy_buffer_resource(device, staging_meshlets_vertices_data_buffer);
        allocator.destroy_buffer_resource(device, staging_meshlets_buffer);
        allocator.destroy_buffer_resource(device, staging_meshlets_triangles_bufer);
        allocator.destroy_buffer_resource(device, staging_meshlets_vertices_buffer);

        let mesh_shader_device = Some(ash::ext::mesh_shader::Device::new(instance, device));

        let punctual_lights_count = punctual_lights.len() as u32;
        let mesh_emitters_count = mesh_emitters.len() as u32;
        let meshlets_count = meshlets.len() as u32;

        Self {
            camera_instances,
            model_matrices,
            materials_buffer,
            meshlets_positions_buffer,
            meshlets_vertices_data_buffer,
            meshlets_buffer,
            meshlets_triangles_buffer,
            meshlets_vertices_buffer,
            punctual_lights: punctual_lights_buffer,
            mesh_emitters: mesh_emitters_buffer,
            meshlets,
            meshlets_count,
            pipeline_data,
            mesh_shader_device,
            camera_names,
            descriptor_pool,
            descriptor_set,
            image_resources,
            punctual_lights_count,
            mesh_emitters_count,
        }
    }

    pub fn render(
        &self,
        device: &ash::Device,
        command_buffer: &vk::CommandBuffer,
        camera_index: usize,
    ) {
        unsafe {
            device.cmd_bind_pipeline(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                *command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let mut pc = PushConstants {
                camera_index: camera_index as u32,
                camera_instances: self.camera_instances.device_address,
                punctual_lights: self.punctual_lights.device_address,
                mesh_emitters: self.mesh_emitters.device_address,
                model_matrices: self.model_matrices.device_address,
                materials: self.materials_buffer.device_address,
                positions: self.meshlets_positions_buffer.device_address,
                meshlets: self.meshlets_buffer.device_address,
                meshlets_triangles: self.meshlets_triangles_buffer.device_address,
                meshlets_vertices: self.meshlets_vertices_buffer.device_address,
                vertices_data: self.meshlets_vertices_data_buffer.device_address,
                punctual_lights_count: self.punctual_lights_count,
                mesh_emitters_count: self.mesh_emitters_count,
                meshlets_count: self.meshlets_count,
            };

            device.cmd_push_constants(
                *command_buffer,
                self.pipeline_data.layout,
                vk::ShaderStageFlags::TASK_EXT
                    | vk::ShaderStageFlags::MESH_EXT
                    | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&pc),
            );

            self.mesh_shader_device
                .as_ref()
                .unwrap()
                .cmd_draw_mesh_tasks(*command_buffer, self.meshlets_count / 32 + 1, 1, 1);
        }
    }

    pub fn images_descriptor_info(&self) -> Vec<vk::DescriptorImageInfo> {
        let mut return_vec = vec![];

        for ir in &self.image_resources {
            return_vec.push(ir.descriptor_info);
        }

        return_vec
    }

    pub fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        unsafe {
            device.queue_wait_idle(*queue).unwrap();
        }
        self.pipeline_data.destroy(device);
        self.pipeline_data =
            PipelineData::new(device, debug_utils, self.image_resources.len() as u32);
    }

    pub fn common_scene_data(&self) -> viewport::CommonSceneData {
        viewport::CommonSceneData {
            camera_instances_addr: self.camera_instances.device_address,
            punctual_lights_addr: self.punctual_lights.device_address,
            mesh_emitters_addr: self.mesh_emitters.device_address,
            materials_addr: self.materials_buffer.device_address,
            model_matrices_addr: self.model_matrices.device_address,
            images_descriptor_info: self.images_descriptor_info(),
            punctual_lights_count: self.punctual_lights_count,
            mesh_emitters_count: self.mesh_emitters_count,
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        for image_resource in self.image_resources {
            allocator.destroy_image_resource(device, image_resource);
        }

        self.pipeline_data.destroy(device);

        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }

        allocator.destroy_buffer_resource(device, self.camera_instances);
        allocator.destroy_buffer_resource(device, self.model_matrices);
        allocator.destroy_buffer_resource(device, self.punctual_lights);
        allocator.destroy_buffer_resource(device, self.mesh_emitters);
        allocator.destroy_buffer_resource(device, self.materials_buffer);
        allocator.destroy_buffer_resource(device, self.meshlets_positions_buffer);
        allocator.destroy_buffer_resource(device, self.meshlets_vertices_data_buffer);
        allocator.destroy_buffer_resource(device, self.meshlets_buffer);
        allocator.destroy_buffer_resource(device, self.meshlets_triangles_buffer);
        allocator.destroy_buffer_resource(device, self.meshlets_vertices_buffer);
    }
}
