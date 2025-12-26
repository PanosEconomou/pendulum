from collections import deque
import taichi as ti
import taichi.math as tm
from taichi import sin, cos, sqrt, exp, log
ti.reset()
ti.init(arch = ti.gpu, fast_math=True)


# Some tweaking is in order
dim     = (1000,800)
upscale = 8
dimH    = (int(upscale*dim[0]),int(upscale*dim[1]))
g       = 1
h       = 5e-2
e       = 1e-3
dx      = (2.5*tm.pi/dimH[1], 2.5*tm.pi/dimH[1]) # (1e-2,1e-2)

# Taichi arrays that will store our vectors
vec     = tm.vec4
a       = ti.Vector.field(4, ti.f32, dimH)
b       = ti.Vector.field(4, ti.f32, dimH)
pixels  = ti.Vector.field(3, ti.f32, dimH)
pixelsL = ti.Vector.field(3, ti.f32, dim)
window  = ti.ui.Window("Double Pendulum", res=dim, fps_limit=60)
canvas  = window.get_canvas()

# Derivative function
@ti.func
def f(a:vec) -> vec:
    # Precalculate some sines ans cosines so that we don't do it twice
    sxy     = sin(  a[0] -   a[1])
    s2x2y   = sin(2*a[0] - 2*a[1])
    c2x2y   = cos(2*a[0] - 2*a[1]) - 3

    # Return the vector b = f(a)
    return  vec(a[2], \
                a[3], \
                (3 * g * sin(a[0]) +   g*sin(a[0] - 2*a[1]) + a[2]*a[2]*s2x2y + 2*a[3]*a[3]*sxy)/c2x2y, \
                (2 * g * sin(a[1]) - 2*g*sin(2*a[0] - a[1]) - a[3]*a[3]*s2x2y - 2*a[2]*a[2]*sxy)/c2x2y) 

# Simple RK4 solver
@ti.func
def step(a:vec, h:ti.f32) -> vec:
    k1 = f(a)
    k2 = f(a + (h/2)*k1)
    k3 = f(a + (h/2)*k2)
    k4 = f(a + h*k3)
    
    return a + (k1 + 2*k2 + 2*k3 + k4)*(h/6)


# Intialization
@ti.kernel
def initialize(dx:float, dy:float, e:float, dimX:int, dimY:int):
    a.fill(0)
    b.fill(0)
    for i,j in a:
        a[i,j][0] = (i-dimX//2            )*dx
        a[i,j][1] = (j-dimY//2            )*dy
        b[i,j][0] = (i-dimX//2 + e/sqrt(2))*dx 
        b[i,j][1] = (j-dimY//2 + e/sqrt(2))*dy

@ti.func
def sigmoid(x:float, k:float):
    return (2/(1+exp(-x/k)) - 1)*7

# Draw the next frame
@ti.kernel
def draw(h:float, norm:float, k:float, colored:bool):
    for i,j in pixels:
        a[i,j] = step(a[i,j], h)
        b[i,j] = step(b[i,j], h)

        if colored:

            pixels[i,j] =  0.5 + 0.5 * ti.Vector([sin(a[i,j][0]/(2*tm.pi)*3.1), ti.sin(a[i,j][1]/(2*tm.pi)*3.1 + 2.1), ti.sin(a[i,j][0]/(2*tm.pi)*1.7 + a[i,j][1]/(2*tm.pi)*2.3)])
        else:
            c = sigmoid((a[i,j][:2] - b[i,j][:2]).norm()/norm, k) # type: ignore
            pixels[i,j] = tm.vec3([c,2*c,5*c])/(7)
   

# Implement some antialiasing
@ti.kernel
def downsample():
    for i, j in pixelsL:
        acc = tm.vec3(0.0)
        for di, dj in ti.ndrange(upscale, upscale):
            acc += pixels[i * upscale + di, j * upscale + dj]
        pixelsL[i, j] = acc / (upscale * upscale)

if __name__ == '__main__':
    initialize(*dx,e,*dimH)

    kmax    = 50
    kmin    = 0.1
    k       = 0.5
    kC      = 0.01
    norm    = e*h
    i       = 0
    every   = 1
    color   = False
    paused  = False
    held_S  = False
    held_R  = False
    held_P  = False

    while window.running:
        if window.is_pressed(ti.GUI.ESCAPE): 
            ti.sync()
            window.destroy()
            break

        if window.is_pressed(ti.GUI.UP):
                if k>kC: k -= kC

        if window.is_pressed(ti.GUI.DOWN):
            if k<kmax-kC: k += kC

        if held_S and not window.is_pressed(ti.GUI.SPACE):
            held_S  = False
            color   = not color
        held_S = window.is_pressed(ti.GUI.SPACE)

        if held_R and not window.is_pressed('r'):
            held_R  = False
            initialize(*dx,e,*dimH)
        held_R = window.is_pressed('r')

        if held_P and not window.is_pressed('p'):
            held_P  = False
            paused  = not paused
        held_P = window.is_pressed('p')
        
        if not paused:
            draw(h, norm, kmin*(kmax/kmin)**k, color)
            if i == 0:
                downsample()
            
            i = (i+1)%every
                
        canvas.set_image(pixelsL)

        window.show()
