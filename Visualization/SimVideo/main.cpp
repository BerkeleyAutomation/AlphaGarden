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
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy5.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint5.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill5.xml");

    WeberPennTreeParameters params_basil = weberpenntree.getTreeParameters("basil6");
    WeberPennTreeParameters params_bokchoy = weberpenntree.getTreeParameters("bokchoy5");
    WeberPennTreeParameters params_mint = weberpenntree.getTreeParameters("mint5");
    WeberPennTreeParameters params_dill = weberpenntree.getTreeParameters("dill5");

    float max_scale_basil = params_basil.Scale;
    float max_leafscale_basil = params_basil.LeafScale;
    float max_scale_bokchoy = params_bokchoy.Scale;
    float max_leafscale_bokchoy = params_bokchoy.LeafScale;
    float max_scale_mint = params_mint.Scale;
    float max_leafscale_mint = params_mint.LeafScale;
    float max_scale_dill = params_dill.Scale;
    
    int Nframes = 100;

    for (int i = 0; i < Nframes; i++)
    {
        // RNG and keep num leaves, branches constant
        // Could also build big tree, then take off branches
        // context.addPatch(make_vec3(0, 0, 0), make_vec2(40, 40), make_SphericalCoord(0, 0), RGB::forestgreen);
        context.addPatch(make_vec3(0, 0, 0), make_vec2(20, 20), make_SphericalCoord(0.f, 0.f), make_RGBcolor(39.6f / 100, 28.6f / 100, 23.4f / 100));

        params_basil.Scale = max_scale_basil * float(i + 1) / float(Nframes);
        params_basil.LeafScale = max_leafscale_basil * float(i + 1) / float(Nframes);
        params_mint.Scale = max_scale_mint * float(i + 1) / float(Nframes);
        params_mint.LeafScale = max_leafscale_mint * float(i + 1) / float(Nframes);
        params_bokchoy.Scale = max_scale_bokchoy * float(i + 1) / float(Nframes);
        params_bokchoy.LeafScale = max_leafscale_bokchoy * float(i + 1) / float(Nframes);
        params_dill.Scale = max_scale_dill * float(i + 1) / float(Nframes);
        // params.Leaves = max_leaves * (i + 1) / Nframes;
        // NcurveRes

        weberpenntree.setTreeParameters("basil6", params_basil);
        weberpenntree.setTreeParameters("bokchoy5", params_bokchoy);
        weberpenntree.setTreeParameters("mint5", params_mint);
        weberpenntree.setTreeParameters("dill5", params_dill);

        weberpenntree.buildTree("basil6", make_vec3(0, 0, 0));
        weberpenntree.buildTree("bokchoy5", make_vec3(1, 0, 0));
        weberpenntree.buildTree("mint5", make_vec3(0, 1, 0));
        weberpenntree.buildTree("dill5", make_vec3(1, 1, 0));

        // float height = 1.f * float(i + 1) / float(Nframes);
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
        vis.clearGeometry();

        context.deletePrimitive(context.getAllUUIDs());
    }
}