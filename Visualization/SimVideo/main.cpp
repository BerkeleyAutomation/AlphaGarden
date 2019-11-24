#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"
// #include "potato.h"

using namespace helios;

int main(void)
{
    Context context;

    WeberPennTree weberpenntree(&context);

    weberpenntree.loadXML("../../tools/hardcoded/basil6.xml");

    WeberPennTreeParameters params = weberpenntree.getTreeParameters("basil6");
    float max_scale = params.Scale;
    float max_leaves = params.Leaves;
    float max_leafscale = params.LeafScale;
    
    int Nframes = 100;

    for (int i = 73; i < Nframes; i++)
    {
        // RNG and keep num leaves, branches constant
        // context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40), make_SphericalCoord(0, 0), RGB::forestgreen);
        context.addPatch(make_vec3(0, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.f, 0.f), make_RGBcolor(39.6f / 100, 28.6f / 100, 23.4f / 100));

        params.Scale = max_scale * float(i + 1) / float(Nframes);
        params.Leaves = max_leaves * float(i + 1) / float(Nframes);
        params.LeafScale = max_leafscale * float(i + 1) / float(Nframes);
        // NcurveRes

        weberpenntree.setTreeParameters("basil6", params);

        uint ID = weberpenntree.buildTree("basil6", make_vec3(0, 0, 0));

        float height = 1.f * float(i + 1) / float(Nframes);
        // potato(make_vec3(5, 0, 0), height, &context);

        Visualizer vis(1200); //Opens a graphics window of width 1200 pixels with default aspect ratio

        vis.buildContextGeometry(&context); //add all geometry in the context to the visualizer

        // vis.setCameraPosition(make_vec3(15, 0, 3), make_vec3(0, 0, 3));
        vis.setCameraPosition(make_vec3(0.1, 0.1, 4), make_vec3(0, 0, 0));
        vis.setCameraFieldOfView(90);
        vis.setLightingModel(Visualizer::LIGHTING_PHONG);
        vis.setLightDirection(make_vec3(0.5, 0.5, 1));

        //vis.plotInteractive();

        char filename[50];
        sprintf(filename, "../videos/trees%03d.jpeg", i);
        vis.plotUpdate();
        struct timespec ts = {0, 100000000L};
        nanosleep(&ts, NULL);
        vis.printWindow(filename);

        vis.closeWindow();
        vis.plotInteractive();
        vis.clearGeometry();

        context.deletePrimitive(context.getAllUUIDs());
    }
}