use amethyst::renderer::rendy::hal::{
    device::Device,
    format::{Aspects, Swizzle},
    image::{
        Anisotropic, Filter, Kind, Layout, PackedColor, SamplerInfo, SubresourceRange, Tiling,
        Usage, ViewCapabilities, ViewKind, WrapMode,
    },
    memory::Dependencies,
    pso::{
        BasePipeline, ComputePipelineDesc, Descriptor, DescriptorSetLayoutBinding,
        DescriptorSetWrite, DescriptorType, EntryPoint, PipelineCreationFlags, ShaderStageFlags,
        Specialization,
    },
    Backend,
};
use amethyst::renderer::rendy::{
    command::{
        CommandBuffer, CommandPool, Compute, ExecutableState, Family, MultiShot, PendingState,
        SimultaneousUse, Submit,
    },
    frame::Frames,
    graph::{
        gfx_acquire_barriers, gfx_release_barriers, BufferAccess, GraphContext, Node, NodeBuffer,
        NodeDesc, NodeImage, NodeSubmittable,
    },
    memory,
    resource::{DescriptorSet, Image, DescriptorSetLayout, Escape, Handle, ImageInfo, ImageViewInfo},
};
use amethyst::{
    prelude::World,
    renderer::{Factory, Format},
};
use failure::Error;

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
        assert!(buffers.is_empty());
        assert!(images.is_empty());

        let mut watcher = rshader::ShaderDirectoryWatcher::new("src/shaders").unwrap();
        let shader_set = rshader::ShaderSet::compute_only(
            &mut watcher,
            shader_source!("../src/shaders", "version", "reproject.comp"),
        )?;

        let module = unsafe {
            factory
                .device()
                .create_shader_module(shader_set.compute())?
        };

        let set_layout = Handle::from(factory.create_descriptor_set_layout(vec![
            DescriptorSetLayoutBinding {
                binding: 0,
                ty: DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                ty: DescriptorType::StorageImage,
                count: 1,
                stage_flags: ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            },
        ])?);

        let pipeline_layout = unsafe {
            factory.device().create_pipeline_layout(
                std::iter::once(set_layout.raw()),
                std::iter::empty::<(ShaderStageFlags, std::ops::Range<u32>)>(),
            )?
        };

        let pipeline = unsafe {
            factory.device().create_compute_pipeline(
                &ComputePipelineDesc {
                    shader: EntryPoint {
                        entry: "main",
                        module: &module,
                        specialization: Specialization::default(),
                    },
                    layout: &pipeline_layout,
                    flags: PipelineCreationFlags::empty(),
                    parent: BasePipeline::None,
                },
                None,
            )?
        };

        unsafe { factory.destroy_shader_module(module) };

        let descriptor_set = factory.create_descriptor_set(set_layout.clone())?;

        let dem: Handle<_> = factory.create_image(
            ImageInfo {
                kind: Kind::D2(1024, 1024, 1, 1),
                levels: 1,
                format: Format::R32Sfloat,
                tiling: Tiling::Linear,
                view_caps: ViewCapabilities::empty(),
                usage: Usage::TRANSFER_DST | Usage::STORAGE | Usage::INPUT_ATTACHMENT,
            },
            memory::Data,
        )?.into();

        let heights: Handle<_> = factory.create_image(
            ImageInfo {
                kind: Kind::D2(1024, 1024, 1, 1),
                levels: 1,
                format: Format::R32Sfloat,
                tiling: Tiling::Linear,
                view_caps: ViewCapabilities::empty(),
                usage: Usage::TRANSFER_DST | Usage::STORAGE | Usage::INPUT_ATTACHMENT,
            },
            memory::Data,
        )?.into();

        let dem_view = factory.create_image_view(
            dem.clone(),
            ImageViewInfo {
                view_kind: ViewKind::D2,
                format: Format::R32Sfloat,
                swizzle: Swizzle::NO,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            },
        )?;
        let heights_view = factory.create_image_view(
            heights.clone(),
            ImageViewInfo {
                view_kind: ViewKind::D2,
                format: Format::R32Sfloat,
                swizzle: Swizzle::NO,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            },
        )?;

        let sampler = factory.create_sampler(SamplerInfo {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            mip_filter: Filter::Nearest,
            wrap_mode: (WrapMode::Clamp, WrapMode::Clamp, WrapMode::Clamp),
            lod_bias: 0.0.into(),
            lod_range: 0.0.into()..1.0.into(),
            comparison: None,
            border: PackedColor(0),
            anisotropic: Anisotropic::Off,
        })?;

        unsafe {
            factory
                .device()
                .write_descriptor_sets(std::iter::once(DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![
                        Descriptor::CombinedImageSampler(
                            dem_view.raw(),
                            Layout::General,
                            sampler.raw(),
                        ),
                        Descriptor::Image(heights_view.raw(), Layout::General),
                    ]
                    .into_iter(),
                }));
        }

        let mut command_pool = factory
            .create_command_pool(family)?
            .with_capability::<Compute>()
            .expect("Graph builder must provide family with Compute capability");
        let initial = command_pool.allocate_buffers(1).remove(0);
        let mut recording = initial.begin(MultiShot(SimultaneousUse), ());
        let mut encoder = recording.encoder();
        encoder.bind_compute_pipeline(&pipeline);
        unsafe {
            encoder.bind_compute_descriptor_sets(
                &pipeline_layout,
                0,
                std::iter::once(descriptor_set.raw()),
                std::iter::empty::<u32>(),
            );

            {
                let (stages, barriers) = gfx_acquire_barriers(ctx, &*buffers, None);
                // log::info!("Acquire {:?} : {:#?}", stages, barriers);
                encoder.pipeline_barrier(stages, Dependencies::empty(), barriers);
            }
            encoder.dispatch(512, 512, 1);
            {
                let (stages, barriers) = gfx_release_barriers(ctx, &*buffers, None);
                // log::info!("Release {:?} : {:#?}", stages, barriers);
                encoder.pipeline_barrier(stages, Dependencies::empty(), barriers);
            }
        }

        let (submit, command_buffer) = recording.finish().submit();

        Ok(ComputeNode {
            set_layout,
            pipeline_layout,
            pipeline,
            descriptor_set,
            command_pool,
            command_buffer,
            submit,
			dem,
			heights,
        })
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

	dem: Handle<Image<B>>,
	heights: Handle<Image<B>>,
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
