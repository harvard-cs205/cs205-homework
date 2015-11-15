from __future__ import division
import sys
import pyopencl as cl
import numpy as np
import pylab
import pdb

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices[:2])
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('label_regions.cl').read()).build(options='')

    host_image = np.load('maze2.npy')
    host_labels = np.empty_like(host_image)
    host_done_flag = np.zeros(1).astype(np.int32)

    gpu_image = cl.Buffer(context, cl.mem_flags.READ_ONLY, host_image.size * 4)
    gpu_labels = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_done_flag = cl.Buffer(context, cl.mem_flags.READ_WRITE, 4)
    #pdb.set_trace()
    # Send to the device, non-blocking
    cl.enqueue_copy(queue, gpu_image, host_image, is_blocking=False)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)])
    print global_size
    width = np.int32(host_image.shape[1])
    height = np.int32(host_image.shape[0])
    halo = np.int32(1)
    
    # Create a local memory per working group that is
    # the size of an int (4 bytes) * (N+2) * (N+2), where N is the local_size
    buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
    gpu_local_memory = cl.LocalMemory(4 * buf_size[0] * buf_size[1])

    # initialize labels
    program.initialize_labels(queue, global_size, local_size,
                              gpu_image, gpu_labels,
                              width, height)

    # while not done, propagate labels
    itercount = 0

    # Show the initial labels
    cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
    pylab.imshow(host_labels)
    pylab.title(itercount)
    pylab.show()

    show_progress = True
    total_time = 0

    while True:
        itercount += 1
        host_done_flag[0] = 0
        print 'iter', itercount
        cl.enqueue_copy(queue, gpu_done_flag, host_done_flag, is_blocking=False)
        prop_exec = program.propagate_labels(queue, global_size, local_size,
                                             gpu_labels, gpu_done_flag,
                                             gpu_local_memory,
                                             width, height,
                                             buf_size[0], buf_size[1],
                                             halo)
        prop_exec.wait()
        elapsed = 1e-6 * (prop_exec.profile.end - prop_exec.profile.start)
        total_time += elapsed
        # read back done flag, block until it gets here
        cl.enqueue_copy(queue, host_done_flag, gpu_done_flag, is_blocking=True)
        if host_done_flag[0] == 0:
            # no changes
            break
        # there were changes, so continue running
        print host_done_flag
        if itercount % 100 == 0 and show_progress:
            cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
            pylab.imshow(host_labels)
            pylab.title(itercount)
            pylab.show()
        if itercount % 10000 == 0:
            print 'Reached maximal number of iterations, aborting'
            sys.exit(0)

    print('Finished after {} iterations, {} ms total, {} ms per iteration'.format(itercount, total_time, total_time / itercount))
    # Show final result
    cl.enqueue_copy(queue, host_labels, gpu_labels, is_blocking=True)
    print 'Found {} regions'.format(len(np.unique(host_labels)) - 1)
    pylab.imshow(host_labels)
    pylab.title(itercount)
    pylab.show()
