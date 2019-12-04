#include "Context.h"
#include "Visualizer.h"
#include "grapevine.h"

using namespace helios;

int main(void)
{
    Context context; //declare the context

    int Nframes = 100;

    for (int i = 0; i < Nframes; i++)
    {

        context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40), make_SphericalCoord(0, 0), RGB::forestgreen);

        float height = 2.f * float(i + 1) / float(Nframes);
        grapevine(make_vec3(5, 0, 0), height, 1, 0 ,&context);

        Visualizer vis(1200); //Opens a graphics window of width 1200 pixels with default aspect ratio

        vis.buildContextGeometry(&context); //add all geometry in the context to the visualizer

        vis.setCameraPosition(make_vec3(10, 0, 3), make_vec3(0, 0, 3));
        vis.setCameraFieldOfView(90);
        vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
        vis.setLightDirection(make_vec3(0.5, 0.5, 1));

        //vis.plotInteractive();

        char filename[50];
        sprintf(filename, "../videos/trees%03d.jpeg", i);
        vis.plotUpdate();
        struct timespec ts = {0, 100000000L};
        nanosleep(&ts, NULL);

        vis.printWindow(filename);

        vis.closeWindow();

        vis.clearGeometry();

        context.deletePrimitive(context.getAllUUIDs());
    }
}