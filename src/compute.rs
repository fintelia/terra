use crate::terrain::quadtree::QuadTree;
use amethyst::ecs::{DispatcherBuilder, WorldExt};
use amethyst::prelude::World;
/// Plugin for Amethyst that renders a terrain.
use amethyst::renderer::{
    bundle::{RenderOrder, RenderPlan, Target},
    submodules::gather::CameraGatherer,
    Factory, RenderPlugin,
};
use failure::Error;
use gfx_hal::buffer::Usage;
use gfx_hal::device::Device;
use gfx_hal::format::Format;
use gfx_hal::pass::Subpass;
use gfx_hal::pso::{
    Comparison, DepthStencilDesc, DepthTest, Descriptor, DescriptorSetLayoutBinding,
    DescriptorSetWrite, DescriptorType, Element, InputAssemblerDesc, PrimitiveRestart,
    ShaderStageFlags, StencilTest, VertexInputRate,
};
use gfx_hal::{Backend, Primitive, QueueType};
use rendy::command::{
    CommandBuffer, CommandPool, Compute, ExecutableState, Family, MultiShot, PendingState, QueueId,
    RenderPassEncoder, SimultaneousUse, Submit,
};
use rendy::frame::Frames;
use rendy::graph::render::{PrepareResult, RenderGroup, RenderGroupBuilder};
use rendy::graph::{
    BufferAccess, BufferId, GraphContext, ImageAccess, ImageId, Node, NodeBuffer, NodeDesc, NodeId,
    NodeImage, NodeSubmittable,
};
use rendy::memory;
use rendy::resource::{DescriptorSet, DescriptorSetLayout, Escape, Handle};

#[derive(Debug, Default)]
pub(crate) struct ComputeNodeDesc;
impl<B: Backend> NodeDesc<B, World> for ComputeNodeDesc {
    type Node = ComputeNode<B>;
    fn buffers(&self) -> Vec<BufferAccess> {
        vec![]
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        queue: usize,
        _aux: &World,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Self::Node, Error> {
        // TODO IMPLEMENT THIS FUNCTION

        
        assert!(buffers.is_empty());
        assert!(images.is_empty());

        // let module = unsafe { BOUNCE_COMPUTE.module(factory) }
        //     .map_err(rendy_core::hal::pso::CreationError::Shader)
        //     .map_err(NodeBuildError::Pipeline)?;

        // let set_layout = Handle::from(
        //     factory
        //         .create_descriptor_set_layout(vec![hal::pso::DescriptorSetLayoutBinding {
        //             binding: 0,
        //             ty: hal::pso::DescriptorType::StorageBuffer,
        //             count: 1,
        //             stage_flags: hal::pso::ShaderStageFlags::COMPUTE,
        //             immutable_samplers: false,
        //         }])
        //         .map_err(NodeBuildError::OutOfMemory)?,
        // );

        // let pipeline_layout = unsafe {
        //     factory
        //         .device()
        //         .create_pipeline_layout(
        //             std::iter::once(set_layout.raw()),
        //             std::iter::empty::<(hal::pso::ShaderStageFlags, std::ops::Range<u32>)>(),
        //         )
        //         .map_err(NodeBuildError::OutOfMemory)?
        // };

        // let pipeline = unsafe {
        //     factory
        //         .device()
        //         .create_compute_pipeline(
        //             &hal::pso::ComputePipelineDesc {
        //                 shader: hal::pso::EntryPoint {
        //                     entry: "main",
        //                     module: &module,
        //                     specialization: hal::pso::Specialization::default(),
        //                 },
        //                 layout: &pipeline_layout,
        //                 flags: hal::pso::PipelineCreationFlags::empty(),
        //                 parent: hal::pso::BasePipeline::None,
        //             },
        //             None,
        //         )
        //         .map_err(NodeBuildError::Pipeline)?
        // };

        // unsafe { factory.destroy_shader_module(module) };

        // let descriptor_set = factory
        //     .create_descriptor_set(set_layout.clone())
        //     .map_err(NodeBuildError::OutOfMemory)?;

        // unsafe {
        //     factory
        //         .device()
        //         .write_descriptor_sets(std::iter::once(hal::pso::DescriptorSetWrite {
        //             set: descriptor_set.raw(),
        //             binding: 0,
        //             array_offset: 0,
        //             descriptors: std::iter::once(hal::pso::Descriptor::Buffer(
        //                 posvelbuff.raw(),
        //                 Some(0)..Some(posvelbuff.size()),
        //             )),
        //         }));
        // }

        // let mut command_pool = factory
        //     .create_command_pool(family)
        //     .map_err(NodeBuildError::OutOfMemory)?
        //     .with_capability::<Compute>()
        //     .expect("Graph builder must provide family with Compute capability");
        // let initial = command_pool.allocate_buffers(1).remove(0);
        // let mut recording = initial.begin(MultiShot(SimultaneousUse), ());
        // let mut encoder = recording.encoder();
        // encoder.bind_compute_pipeline(&pipeline);
        // unsafe {
        //     encoder.bind_compute_descriptor_sets(
        //         &pipeline_layout,
        //         0,
        //         std::iter::once(descriptor_set.raw()),
        //         std::iter::empty::<u32>(),
        //     );

        //     {
        //         let (stages, barriers) = gfx_acquire_barriers(ctx, &*buffers, None);
        //         log::info!("Acquire {:?} : {:#?}", stages, barriers);
        //         encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
        //     }
        //     encoder.dispatch(QUADS, 1, 1);

        //     {
        //         let (stages, barriers) = gfx_release_barriers(ctx, &*buffers, None);
        //         log::info!("Release {:?} : {:#?}", stages, barriers);
        //         encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
        //     }
        // }

        // let (submit, command_buffer) = recording.finish().submit();

        // Ok(GravBounce {
        //     set_layout,
        //     pipeline_layout,
        //     pipeline,
        //     descriptor_set,
        //     // buffer_view,
        //     command_pool,
        //     command_buffer,
        //     submit,
        // })
        unimplemented!()
    }
}
#[derive(Debug)]
pub(crate) struct ComputeNode<B: Backend> {
    set_layout: Handle<DescriptorSetLayout<B>>,
    pipeline_layout: B::PipelineLayout,
    pipeline: B::ComputePipeline,

    descriptor_set: Escape<DescriptorSet<B>>,

    command_pool: CommandPool<B, Compute>,
    command_buffer:
        CommandBuffer<B, Compute, PendingState<ExecutableState<MultiShot<SimultaneousUse>>>>,
    submit: Submit<B, SimultaneousUse>,
}
impl<'a, B: Backend> NodeSubmittable<'a, B> for ComputeNode<B> {
    type Submittable = &'a Submit<B, SimultaneousUse>;
    type Submittables = &'a [Submit<B, SimultaneousUse>];
}
impl<B: Backend> Node<B, World> for ComputeNode<B> {
    type Capability = Compute;
    type Desc = ComputeNodeDesc;
    fn run(
        &mut self,
        ctx: &GraphContext<B>,
        factory: &Factory<B>,
        aux: &World,
        frames: &Frames<B>,
    ) -> &[Submit<B, SimultaneousUse>] {
        std::slice::from_ref(&self.submit)
    }

    unsafe fn dispose(mut self, factory: &mut Factory<B>, aux: &World) {
        drop(self.submit);
        self.command_pool
            .free_buffers(Some(self.command_buffer.mark_complete()));
        factory.destroy_command_pool(self.command_pool);
        factory.destroy_compute_pipeline(self.pipeline);
        factory.destroy_pipeline_layout(self.pipeline_layout);
    }
}
