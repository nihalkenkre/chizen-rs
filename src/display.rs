use std::ffi::CStr;

use ash::vk::{self};

use crate::{
    ui_imgui,
    vulkan_objects::{self, AllocatorTrait},
};

struct PipelineData {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

#[derive(Debug, Default, Clone, Copy)]
struct PushConstants {
    pub pos_offset: [f32; 2],
    pub zoom_level: f32,
}

unsafe impl bytemuck::NoUninit for PushConstants {}

impl PipelineData {
    pub fn new(device: &ash::Device, debug_utils: &vulkan_objects::DebugUtils) -> Self {
        let spirv_data = std::fs::read(
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string()
                + ("/shaders/display.slang.spv"),
        )
        .unwrap();

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
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("vertex_main\0".as_bytes()).unwrap()),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader_mod)
                .name(CStr::from_bytes_with_nul("fragment_main\0".as_bytes()).unwrap()),
        ];
        let vibd =
            [vk::VertexInputBindingDescription::default().stride((size_of::<f32>() * 4) as u32)];

        let viad = [
            vk::VertexInputAttributeDescription::default().format(vk::Format::R32G32_SFLOAT),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset((size_of::<f32>() * 2) as u32),
        ];
        let vis_ci = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&vibd)
            .vertex_attribute_descriptions(&viad);
        let ias_ci = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
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

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let descriptor_set_layout;
        unsafe {
            descriptor_set_layout = device.create_descriptor_set_layout(&dsl_ci, None).unwrap();
        }

        let pc_ranges = [vk::PushConstantRange::default()
            .size(size_of::<PushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];

        let dsls = [descriptor_set_layout];
        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&pc_ranges)
            .set_layouts(&dsls);

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
            .vertex_input_state(&vis_ci)
            .input_assembly_state(&ias_ci)
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
            debug_utils.set_object_name(pipeline, "display pipeline");
            debug_utils.set_object_name(layout, "display pipeline layout");
            debug_utils.set_object_name(descriptor_set_layout, "display descriptor set layout");
        }

        Self {
            pipeline,
            layout,
            descriptor_set_layout,
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

pub struct Display {
    frame_objects: vulkan_objects::FrameObjects,
    acquire_signal_semaphores: Vec<vk::Semaphore>,
    present_wait_semaphores: Vec<vk::Semaphore>,
    max_frames_in_flight: usize,

    pipeline_data: PipelineData,

    geometry_buffer_resource: vulkan_objects::BufferResource,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl Display {
    pub fn new(
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        queue_family_indices: &[u32],
        extent: &vk::Extent2D,
        max_frames_in_flight: usize,
        swapchain_image_count: usize,
        frame_objects_queue_family_index: u32,
        debug_utils: &vulkan_objects::DebugUtils,
        window: &winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        data_helper: &mut vulkan_objects::DataHelper,
        final_render_target: &vulkan_objects::ImageResource,
        name: &str,
    ) -> Self {
        let mut present_wait_semaphores = Vec::new();
        let mut acquire_signal_semaphores = Vec::new();
        let sem_ci = vk::SemaphoreCreateInfo::default();

        unsafe {
            for _ in 0..max_frames_in_flight {
                acquire_signal_semaphores.push(device.create_semaphore(&sem_ci, None).unwrap());
            }

            for _ in 0..swapchain_image_count {
                present_wait_semaphores.push(device.create_semaphore(&sem_ci, None).unwrap());
            }
        }

        if cfg!(debug_assertions) {
            for (index, object) in present_wait_semaphores.iter().enumerate() {
                let name =
                    name.to_owned() + " present wait semaphore " + index.to_string().as_str();
                debug_utils.set_object_name(*object, name.as_str());
            }

            for (index, object) in acquire_signal_semaphores.iter().enumerate() {
                let name =
                    name.to_owned() + " acquire signal semaphore " + index.to_string().as_str();
                debug_utils.set_object_name(*object, name.as_str());
            }
        }

        let pipeline_data = PipelineData::new(device, debug_utils);

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .descriptor_count(1)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)];

        let dsp_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);

        let descriptor_pool;
        unsafe {
            descriptor_pool = device.create_descriptor_pool(&dsp_ci, None).unwrap();
        }

        let dsls = [pipeline_data.descriptor_set_layout];

        let ds_ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&dsls);

        let descriptor_set;

        unsafe {
            descriptor_set = device.allocate_descriptor_sets(&ds_ai).unwrap()[0];
        }

        let verts: [f32; 24] = [
            -1.0, 1.0, 0.0, 1.0, // top-left
            -1.0, -1.0, 0.0, 0.0, // bottom-left
            1.0, -1.0, 1.0, 0.0, // bottom-right
            -1.0, 1.0, 0.0, 1.0, // top-left
            1.0, -1.0, 1.0, 0.0, // bottom-right
            1.0, 1.0, 1.0, 1.0, // top-right
        ];

        let staging_geometry_buffer = allocator.create_buffer_on_host_with_data(
            device,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &verts,
            debug_utils,
            None,
            "staging display geometry",
        );

        let geometry_buffer_resource = allocator.create_buffer_on_device(
            device,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            (verts.len() * size_of_val(&verts[0])) as vk::DeviceSize,
            debug_utils,
            None,
            "display geometry",
        );

        data_helper.record_batch(device);
        data_helper.copy_buffer_to_buffer(
            device,
            staging_geometry_buffer.vk_buffer(),
            geometry_buffer_resource.vk_buffer(),
            staging_geometry_buffer.data_size,
        );
        data_helper.submit_batch(device, None, None, None);

        if cfg!(debug_assertions) {
            debug_utils.set_object_name(descriptor_pool, "display descriptor pool");
            debug_utils.set_object_name(descriptor_set, "display descriptor set");
        }

        let image_infos = [final_render_target.descriptor_info];
        let write_desc_set = vk::WriteDescriptorSet::default()
            .image_info(&image_infos)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(descriptor_set);

        unsafe {
            device.update_descriptor_sets(&[write_desc_set], &[]);
        }

        allocator.destroy_buffer_resource(device, staging_geometry_buffer);

        Self {
            frame_objects: vulkan_objects::FrameObjects::new(
                device,
                frame_objects_queue_family_index,
                max_frames_in_flight,
                debug_utils,
            ),
            present_wait_semaphores,
            acquire_signal_semaphores,
            max_frames_in_flight,
            pipeline_data,
            descriptor_pool,
            descriptor_set,
            geometry_buffer_resource,
        }
    }

    pub fn update_final_render_target_desc(
        &self,
        device: &ash::Device,
        final_render_target_desc: vk::DescriptorImageInfo,
    ) {
        let image_infos = [final_render_target_desc];
        let write_desc_set = vk::WriteDescriptorSet::default()
            .image_info(&image_infos)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(self.descriptor_set);

        unsafe {
            device.update_descriptor_sets(&[write_desc_set], &[]);
        }
    }

    pub fn render(
        &mut self,
        device: &ash::Device,
        swapchain: &vulkan_objects::Swapchain,
        queue: &vk::Queue,
        extent: &vk::Extent2D,
        ui: &mut ui_imgui::UI,
        window: &winit::window::Window,
        data_helper: &vulkan_objects::DataHelper,
        pos_offset: [f32; 2],
        zoom_level: f32,
    ) {
        let cmd_buff = self.frame_objects.get_command_buffer();
        let frame_sem = self.frame_objects.get_semaphore();
        let frame_sem_value = self.frame_objects.get_frame_sem_value();
        let frame_in_flight = self.frame_objects.get_frame_in_flight();

        let values = [frame_sem_value];
        let acq_wait_sem = [*frame_sem];

        // wait for previous frame to get done before acquiring the presentation image index.
        let acq_wait_info = vk::SemaphoreWaitInfo::default()
            .values(&values)
            .semaphores(&acq_wait_sem);

        let acq_info = vk::AcquireNextImageInfoKHR::default()
            .device_mask(0x1)
            .timeout(u64::MAX)
            .semaphore(*self.acquire_signal_semaphores.get(frame_in_flight).unwrap())
            .swapchain(swapchain.swapchain);

        let image_index;

        unsafe {
            device.wait_semaphores(&acq_wait_info, u64::MAX).unwrap();
            match swapchain.loader.acquire_next_image2(&acq_info) {
                Ok(id) => {
                    if !id.1 {
                        image_index = id.0;
                    } else {
                        eprintln!("Swapchain is suboptimal");
                        return;
                    }
                }
                Err(err) => {
                    eprintln!("{err}");
                    return;
                }
            }
        }

        // change the presentation image layout for color attachment, clear the image, setup rendering info 
        let cmd_buff_bi = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        let pre_render_image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(*swapchain.images.get(image_index as usize).unwrap())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )];
        let pre_render_dep_info =
            vk::DependencyInfo::default().image_memory_barriers(&pre_render_image_memory_barriers);

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(*swapchain.image_views.get(image_index as usize).unwrap())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.0, 0.0, 1.0],
                },
            })];

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D::default().extent(*extent))
            .layer_count(1)
            .color_attachments(&color_attachments);

        let viewports = [vk::Viewport::default()
            .width(1920 as f32)
            .height(1080 as f32)
            .min_depth(0f32)
            .max_depth(1f32)];

        let scissors = [vk::Rect2D::default().extent(*extent)];

        // presentation image to present_src_khr
        let post_render_image_memory_barriers = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(swapchain.images[image_index as usize])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1)
                    .level_count(1),
            )];
        let post_render_dep_info =
            vk::DependencyInfo::default().image_memory_barriers(&post_render_image_memory_barriers);

        // wait for acquire signal semaphore to indicate that the image is acquired and ready to be drawn to.
        // wait for data_helper semaphore, which will signal the end of any transfer that this submit might be dependent on.
        let wait_sem_infos = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    *self
                        .acquire_signal_semaphores
                        .get(frame_in_flight as usize)
                        .unwrap(),
                )
                .stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(data_helper.semaphore)
                .value(data_helper.semaphore_value)
                .stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER),
        ];

        let cmd_buff_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(*cmd_buff)];

        // signal the present wait semaphore, tell the presentation engine, the image is ready to present.
        // signal the FrameObjects semaphore, with an incremented value, we can wait on this value - enabled by calling next_frame() - at the start of the next render call.
        let sig_sem_infos = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(
                    *self
                        .present_wait_semaphores
                        .get(image_index as usize)
                        .unwrap(),
                )
                .stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(*frame_sem)
                .value(frame_sem_value + 1)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
        ];

        let submit_infos = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_sem_infos)
            .command_buffer_infos(&cmd_buff_infos)
            .signal_semaphore_infos(&sig_sem_infos)];

        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_wait_semaphores = [*self
            .present_wait_semaphores
            .get(image_index as usize)
            .unwrap()];

        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .wait_semaphores(&present_wait_semaphores);

        let buffers = [self.geometry_buffer_resource.descriptor_info.buffer];
        let pc = PushConstants {
            pos_offset,
            zoom_level,
        };

        // This unsafe block has all the 'action' parts of the rendering sequence, they reference the data created above.
        unsafe {
            device
                .begin_command_buffer(*cmd_buff, &cmd_buff_bi)
                .unwrap();
            device.cmd_pipeline_barrier2(*cmd_buff, &pre_render_dep_info);
            device.cmd_begin_rendering(*cmd_buff, &rendering_info);

            device.cmd_set_scissor_with_count(*cmd_buff, &scissors);
            device.cmd_set_viewport_with_count(*cmd_buff, &viewports);
            device.cmd_bind_pipeline(
                *cmd_buff,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                *cmd_buff,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_data.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                *cmd_buff,
                self.pipeline_data.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&pc),
            );
            device.cmd_bind_vertex_buffers(*cmd_buff, 0, &buffers, &[0]);
            device.cmd_draw(*cmd_buff, 6, 1, 0, 0);
        }

        ui.render(window, cmd_buff);

        unsafe {
            device.cmd_end_rendering(*cmd_buff);
            device.cmd_pipeline_barrier2(*cmd_buff, &post_render_dep_info);
            device.end_command_buffer(*cmd_buff).unwrap();
            device
                .queue_submit2(*queue, &submit_infos, vk::Fence::null())
                .unwrap();

            // this because queue_present fails and unwrap would panic.
            // queue present failure might be due to window resize, and can be ignored
            match swapchain.loader.queue_present(*queue, &present_info) {
                Ok(_) => {}
                Err(err) => {
                    eprintln!("{err}");
                }
            }

            // This is required to prevent GPU from stalling.
            // Compute queue writing to final_render_target, Graphics queue reading from it.
            // Wierd
            // and max_frames_in_flight can be 1
            device.queue_wait_idle(*queue).unwrap();
        }

        self.frame_objects.next_frame();
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        unsafe {
            self.frame_objects.destroy(device);
            self.pipeline_data.destroy(device);
            allocator.destroy_buffer_resource(device, self.geometry_buffer_resource);

            device.destroy_descriptor_pool(self.descriptor_pool, None);

            for semaphore in self.acquire_signal_semaphores {
                device.destroy_semaphore(semaphore, None);
            }

            for semaphore in self.present_wait_semaphores {
                device.destroy_semaphore(semaphore, None);
            }
        }
    }
}
