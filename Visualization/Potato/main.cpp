#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"
#include "potato.h"

using namespace helios;

int main(void)
{
    Context context; //declare the context

    //create an instance of the WeberPennTree class, which we will call "weberpenntree"
    WeberPennTree weberpenntree(&context);

    // WeberPennTreeParameters params = weberpenntree.getTreeParameters("Orange");

    int Nframes = 100;

    for (int i = 0; i < Nframes; i++)
    {

        context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40), make_SphericalCoord(0, 0), RGB::forestgreen);

        // params.Scale = 8.f * float(i + 1) / float(Nframes);
        // params.Leaves = 20 * float(i + 1) / float(Nframes);
        // params.LeafScale = 0.2 * float(i + 1) / float(Nframes);

        // weberpenntree.setTreeParameters("Orange", params);

        // uint ID = weberpenntree.buildTree("Orange", make_vec3(0, 0, 0));

        float height = 2.f * float(i + 1) / float(Nframes);
        potato(make_vec3(5, 0, 0), height, &context);

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