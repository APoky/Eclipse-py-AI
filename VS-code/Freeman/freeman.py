import numpy as np
import pyopencl as cl
import cv2

# ----- OpenCL kernel: Sobel (gx,gy) -> magnitude -> threshold -> binary -----
KERNEL = r"""
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void sobel_threshold(__read_only image2d_t src, __write_only image2d_t dst, float thresh) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int2 coords = (int2)(x,y);
    int width = get_image_width(src);
    int height = get_image_height(src);

    if (x < 1 || y < 1 || x >= width-1 || y >= height-1) {
        float4 out = (float4)(0.0f,0.0f,0.0f,1.0f);
        write_imagef(dst, coords, out);
        return;
    }

    // Read 3x3 grayscale neighborhood (assumes single-channel stored in r)
    float p00 = read_imagef(src, sampler, (int2)(x-1,y-1)).x;
    float p10 = read_imagef(src, sampler, (int2)(x,y-1)).x;
    float p20 = read_imagef(src, sampler, (int2)(x+1,y-1)).x;

    float p01 = read_imagef(src, sampler, (int2)(x-1,y)).x;
    float p11 = read_imagef(src, sampler, (int2)(x,y)).x;
    float p21 = read_imagef(src, sampler, (int2)(x+1,y)).x;

    float p02 = read_imagef(src, sampler, (int2)(x-1,y+1)).x;
    float p12 = read_imagef(src, sampler, (int2)(x,y+1)).x;
    float p22 = read_imagef(src, sampler, (int2)(x+1,y+1)).x;

    // Sobel kernels
    float gx = -p00 + p20 - 2.0f*p01 + 2.0f*p21 - p02 + p22;
    float gy = -p00 - 2.0f*p10 - p20 + p02 + 2.0f*p12 + p22;

    float mag = sqrt(gx*gx + gy*gy);

    float outval = mag > thresh ? 1.0f : 0.0f;
    float4 out = (float4)(outval, outval, outval, 1.0f);
    write_imagef(dst, coords, out);
}
"""

# ----- Freeman (Moore neighbor) 8-direction chain-code tracer -----
# Direction mapping: 0=E,1=NE,2=N,3=NW,4=W,5=SW,6=S,7=SE
DIRS = [
    (0, 1),   # 0 E
    (-1, 1),  # 1 NE
    (-1, 0),  # 2 N
    (-1, -1), # 3 NW
    (0, -1),  # 4 W
    (1, -1),  # 5 SW
    (1, 0),   # 6 S
    (1, 1),   # 7 SE
]

def freeman_chain_codes(binary):
    """
    binary : 2D numpy array (uint8 or bool), foreground==1, background==0
    Returns list of contours: each contour is dict { 'start':(r,c), 'coords':[(r,c),...], 'codes':[d0,d1,...] }
    """
    H, W = binary.shape
    visited = np.zeros_like(binary, dtype=np.uint8)
    contours = []

    def in_bounds(r,c):
        return 0 <= r < H and 0 <= c < W

    # helper to get index of direction vector (dr,dc) in DIRS
    dir_to_idx = {d:i for i,d in enumerate(DIRS)}

    # iterate to find unvisited foreground pixels (potential starts)
    for r0 in range(H):
        for c0 in range(W):
            if binary[r0,c0] == 0 or visited[r0,c0]:
                continue

            # Check if this pixel is a border pixel (has at least one background neighbor).
            is_border = False
            for dr,dc in DIRS:
                rr,cc = r0+dr, c0+dc
                if not in_bounds(rr,cc) or binary[rr,cc] == 0:
                    is_border = True
                    break
            if not is_border:
                # interior pixel; we can mark visited and skip actual contour tracing now
                # To keep algorithm simple, skip interior pixels — they will be covered by contour starting at boundary
                visited[r0,c0] = 1
                continue

            # Moore neighbor tracing start
            start = (r0, c0)
            # backtrack point b = start + W (i.e., (r0, c0-1))
            b = (r0, c0-1)
            p = start
            c = b

            chain_coords = [p]
            chain_codes = []

            # To avoid infinite loops on malformed masks, set a step cap
            max_steps = H * W + 10
            steps = 0
            while True:
                steps += 1
                if steps > max_steps:
                    # give up on this contour
                    break

                # compute index of vector (c - p) in DIRS
                vec = (c[0] - p[0], c[1] - p[1])
                start_idx = dir_to_idx.get(vec, None)
                if start_idx is None:
                    # sometimes c lies out-of-bounds or vector not in DIRS; fall back to 0
                    start_idx = 0
                # search neighbors starting from (start_idx+1) mod 8 clockwise
                found = False
                for s in range(1, 9):
                    idx = (start_idx + s) % 8
                    dr, dc = DIRS[idx]
                    rr = p[0] + dr
                    cc = p[1] + dc
                    if not in_bounds(rr,cc):
                        # treat out of bounds as background
                        continue
                    if binary[rr,cc] != 0:
                        # found next boundary pixel q
                        q = (rr,cc)
                        chain_codes.append(idx)
                        chain_coords.append(q)
                        c = p
                        p = q
                        found = True
                        break
                if not found:
                    # no neighbor found (isolated pixel?) break
                    break

                if p == start and c == b:
                    # completed closed contour
                    break

            # mark visited pixels along this contour
            for (rr,cc) in chain_coords:
                if in_bounds(rr,cc):
                    visited[rr,cc] = 1

            if len(chain_codes) > 0:
                contours.append({
                    'start': start,
                    'coords': chain_coords,
                    'codes': chain_codes
                })

    return contours

# ----- Utility: Run OpenCL sobel+threshold -----
def sobel_threshold_opencl(gray_img, threshold=0.2):
    # gray_img: float32 in [0..1], shape (H,W)
    H, W = gray_img.shape
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Convert to RGBA float image (4 channels) for image2d_t convenience
    rgba = np.empty((H, W, 4), dtype=np.float32)
    rgba[...,0] = gray_img
    rgba[...,1] = gray_img
    rgba[...,2] = gray_img
    rgba[...,3] = 1.0

    mf = cl.mem_flags
    fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    src_img = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, shape=(W,H), hostbuf=rgba)
    dst_img = cl.Image(ctx, mf.WRITE_ONLY, fmt, shape=(W,H))

    prg = cl.Program(ctx, KERNEL).build()
    prg.sobel_threshold(queue, (W, H), None, src_img, dst_img, np.float32(threshold))

    out = np.empty_like(rgba)
    origin = (0,0,0)
    region = (W, H, 1)
    cl.enqueue_copy(queue, out, dst_img, origin=origin, region=region)
    # foreground channel is out[...,0]
    binary = (out[...,0] > 0.5).astype(np.uint8)
    return binary

# ----- Example usage -----
if __name__ == '__main__':
    # read image as grayscale, normalize to [0..1]
    img = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit("Place an 'example.png' next to script or change filename.")
    gray = img.astype(np.float32) / 255.0

    # GPU: compute edges and threshold -> binary
    #binary = sobel_threshold_opencl(gray, threshold=0.15)

    # Optional: morphological close/open or dilate to join small gaps (use cv2 or implement in OpenCL)
    # binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))

    # CPU: Freeman chain-code tracing
    #contours = freeman_chain_codes(binary)
    contours = freeman_chain_codes(gray)

    print(f"Found {len(contours)} contours.")
    for i,c in enumerate(contours):
        print(f"Contour {i}: start={c['start']}, length={len(c['codes'])}")
        # Example show chain codes and first few coords
        print("Codes (first 20):", c['codes'][:20])
        print("Coords (first 5):", c['coords'][:5])

    # visualize overlay (optional)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cont in contours:
        pts = np.array([(c[1], c[0]) for c in cont['coords']], dtype=np.int32)  # (x,y)
        if pts.shape[0] > 1:
            cv2.polylines(vis, [pts], isClosed=True, color=(0,0,255), thickness=1)
    cv2.imshow('contours', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
