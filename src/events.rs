use std::path::PathBuf;

use ash::vk;

#[derive(Debug)]
pub enum UserEvent {
    FileOpen(PathBuf),
    ReloadShaders,
    StartRender,
    RenderStarted,
    RenderStopped,
    RenderExtentChanged(vk::Extent2D),
}
