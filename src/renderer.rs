use std::{
    io::Write,
    sync::{Arc, atomic::AtomicBool},
    u64,
};

use ash::vk::{self};
use winit::event_loop::EventLoopProxy;

use crate::{
    events::{self, UserEvent},
    renderer::scene::SceneTrait,
    renderer::scene::SceneType,
    viewport,
    vulkan_objects::{self, AllocatorTrait},
};

mod scene;

/// The raytracer. It sets up the semaphores, command buffers etc, and calls the render function on the scene object
pub struct Renderer {
    pub compute_helper: vulkan_objects::DataHelper,

    /// Accumalation target for per sample output.
    accum_target: vulkan_objects::ImageResource,
    event_proxy: EventLoopProxy<UserEvent>,

    frame_objects: vulkan_objects::FrameObjects,

    /// What we are rendering.
    scene: scene::SceneType,
}

impl Renderer {
    pub fn new(
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        extent: &vk::Extent2D,
        queue_family_indices: &[u32],
        compute_queue: &vk::Queue,
        compute_queue_family_index: u32,
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
        event_proxy: EventLoopProxy<events::UserEvent>,
        max_frames_in_flight: usize,
    ) -> Self {
        let accum_target = allocator.create_image(
            device,
            extent,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            queue_family_indices,
            debug_utils,
            "render accum target",
        );

        data_helper.record_batch(device);
        data_helper.change_image_layout(
            device,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::ImageAspectFlags::COLOR,
            &accum_target.image,
        );
        data_helper.submit_batch(device, None, None, None);

        let frame_objects = vulkan_objects::FrameObjects::new(
            device,
            compute_queue_family_index,
            max_frames_in_flight,
            debug_utils,
        );

        let compute_helper = vulkan_objects::DataHelper::new(
            device,
            &compute_queue,
            compute_queue_family_index,
            debug_utils,
        );

        Self {
            accum_target,
            event_proxy,
            frame_objects,
            scene: SceneType::default(),
            compute_helper,
        }
    }

    pub fn create_scene(
        &mut self,
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
        common_scene_data: &viewport::CommonSceneData,
    ) {
        unsafe {
            device.device_wait_idle().unwrap();
        }

        std::mem::take(&mut self.scene).destroy(device, allocator);

        self.scene = SceneType::World(scene::Scene::new(
            gltf,
            buffers,
            images,
            instance,
            device,
            allocator,
            queue_family_indices,
            compute_queue,
            raytracing_properties,
            raytracing_loader,
            scratch_buffer_alignment,
            debug_utils,
            data_helper,
            &mut self.compute_helper,
            common_scene_data,
        ));

        self.scene
            .update_accum_target_desc(device, self.accum_target.descriptor_info);
    }

    pub fn reload_shaders(
        &mut self,
        device: &ash::Device,
        queue: &vk::Queue,
        final_render_target: vk::DescriptorImageInfo,
        debug_utils: &vulkan_objects::DebugUtils,
    ) {
        self.scene.reload_shaders(
            device,
            queue,
            self.accum_target.descriptor_info,
            final_render_target,
            debug_utils,
        );
    }

    pub fn start(
        &mut self,
        device: &ash::Device,
        final_render_target: vk::DescriptorImageInfo,
        queue: vk::Queue,
        render_extent: vk::Extent2D,
        num_samples: u32,
        render_mode: u32,
        camera_index: u32,
        bounce_count: u32,
        random_states: vk::DeviceAddress,
        common_scene_data: viewport::CommonSceneData,
        should_stop: Arc<AtomicBool>,
    ) {
        print!("Rendering...");
        std::io::stdout().flush().unwrap();

        // clear the accumalation image. Clean slate.
        self.compute_helper.record_batch(device);
        self.compute_helper
            .clear_image(device, &self.accum_target.image);
        self.compute_helper.submit_batch(device, None, None, None);

        // foreach sample, wait for the previous sample to finish, render, repeat.
        for s in 1..num_samples + 1 {
            if should_stop.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            let command_buffer = self.frame_objects.get_command_buffer();
            let frame_sem = self.frame_objects.get_semaphore();
            let frame_sem_value = self.frame_objects.get_frame_sem_value();
            let frame_in_flight = self.frame_objects.get_frame_in_flight();

            let wait_values = [frame_sem_value];
            let wait_sems = [*frame_sem];

            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&wait_sems)
                .values(&wait_values);

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            let command_buffer_infos =
                [vk::CommandBufferSubmitInfo::default().command_buffer(*command_buffer)];
            let sig_sem_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(*frame_sem)
                .value(frame_sem_value + 1)
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];

            let submits = [vk::SubmitInfo2::default()
                .command_buffer_infos(&command_buffer_infos)
                .signal_semaphore_infos(&sig_sem_infos)];

            unsafe {
                device.wait_semaphores(&wait_info, u64::MAX).unwrap();
                device
                    .begin_command_buffer(*command_buffer, &begin_info)
                    .unwrap();

                self.scene.render(
                    device,
                    random_states,
                    self.accum_target.descriptor_info,
                    final_render_target,
                    render_extent,
                    *command_buffer,
                    s,
                    render_mode,
                    camera_index,
                    bounce_count,
                    &common_scene_data,
                );

                device.end_command_buffer(*command_buffer).unwrap();

                device
                    .queue_submit2(queue, &submits, vk::Fence::null())
                    .unwrap();

                // This is required to prevent GPU from stalling.
                // Compute queue writing to final_render_target, Graphics queue reading from it.
                // Wierd
                // and so max_frames_in_flight can be 1
                device.queue_wait_idle(queue).unwrap();
            }

            self.frame_objects.next_frame();
        }

        println!("done");

        should_stop.store(false, std::sync::atomic::Ordering::Relaxed);
        self.event_proxy
            .send_event(events::UserEvent::RenderStopped)
            .unwrap();
    }

    pub fn recreate_accum_target(
        &mut self,
        device: &ash::Device,
        allocator: &mut vulkan_objects::AllocatorType,
        extent: &vk::Extent2D,
        queue_family_indices: &[u32],
        debug_utils: &vulkan_objects::DebugUtils,
        data_helper: &mut vulkan_objects::DataHelper,
    ) {
        allocator.destroy_image_resource(device, std::mem::take(&mut self.accum_target));
        self.accum_target = allocator.create_image(
            device,
            extent,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
            queue_family_indices,
            debug_utils,
            "render accum target",
        );

        data_helper.record_batch(device);
        data_helper.change_image_layout(
            device,
            vk::PipelineStageFlags2::TOP_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::empty(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::ImageAspectFlags::COLOR,
            &self.accum_target.image,
        );
        data_helper.submit_batch(device, None, None, None);

        // tell the scene about the new accum target 
        self.scene
            .update_accum_target_desc(device, self.accum_target.descriptor_info);
    }

    pub fn update_final_render_target_desc(
        &self,
        device: &ash::Device,
        final_render_target_desc: vk::DescriptorImageInfo,
    ) {
        self.scene
            .update_final_render_target_desc(device, final_render_target_desc);
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut vulkan_objects::AllocatorType) {
        allocator.destroy_image_resource(device, std::mem::take(&mut self.accum_target));
        self.frame_objects.destroy(device);
        self.compute_helper.destroy(device);

        if let scene::SceneType::World(scene) = &mut self.scene {
            scene.destroy(device, allocator);
        }
    }
}
