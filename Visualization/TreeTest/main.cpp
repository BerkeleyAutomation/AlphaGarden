#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"

using namespace helios;

int main(void)
{
    Context context; //declare the context

    //create an instance of the WeberPennTree class, which we will call "weberpenntree"
    WeberPennTree weberpenntree(&context);
    //Create an almond tree at the point (0,0,0)
    // uint ID_almond = weberpenntree.buildTree( "Almond", make_vec3(0,10,0) );
    //Retrieve UUIDs for context primitives making up the almond tree's leaves
    // std::vector<uint> leafUUIDs_almond = weberpenntree.getLeafUUIDs( ID_almond );
    // uint ID_avocado = weberpenntree.buildTree( "Avocado", make_vec3(10,0,0) );

    weberpenntree.loadXML("../../tools/hardcoded/basil1.xml");
    weberpenntree.loadXML("../../tools/hardcoded/basil2.xml");
    weberpenntree.loadXML("../../tools/hardcoded/basil3.xml");
    weberpenntree.loadXML("../../tools/hardcoded/basil4.xml");
    weberpenntree.loadXML("../../tools/hardcoded/basil6.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint1.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint2.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint3.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint4.xml");
    weberpenntree.loadXML("../../tools/hardcoded/mint5.xml");
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy1.xml");
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy2.xml");
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy3.xml");
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy4.xml");
    weberpenntree.loadXML("../../tools/hardcoded/bokchoy5.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill1.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill2.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill3.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill4.xml");
    weberpenntree.loadXML("../../tools/hardcoded/dill5.xml");
    Visualizer vis(1200); //Opens a graphics window of width 1200 pixels with default aspect ratio
    // vis.addSkyDomeByCenter(100, make_vec3(0, 0, 0), 20, "plugins/visualizer/textures/SkyDome_clouds.jpg", 0);
    // context.addPatch(make_vec3(0, 0, 0), make_vec2(10,10), make_SphericalCoord(0.f, 0.f), "plugins/visualizer/textures/dirt.jpg");
    // context.addPatch(make_vec3(1, 1, 0), make_vec2(5,5), make_SphericalCoord(0.f, 0.f), "plugins/visualizer/textures/marble_white.jpg");
    // context.addPatch(make_vec3(1, 1, 0), make_vec2(5,5), make_SphericalCoord(0.f, 0.f), "../images/soil.jpg");
    context.addPatch(make_vec3(1, 1, 0), make_vec2(5,5), make_SphericalCoord(0.f, 0.f), make_RGBcolor(39.6f/100,28.6f/100,23.4f/100));

    weberpenntree.buildTree("bokchoy5", make_vec3(2, 1.1, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(2, 0.8, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(2, 0.5, 0));
    weberpenntree.buildTree("bokchoy2", make_vec3(2, 0, 0));
    weberpenntree.buildTree("mint3", make_vec3(1.9, 1.7, 0));
    weberpenntree.buildTree("basil3", make_vec3(1.8, 1.8, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(1.8, 1.6, 0));
    weberpenntree.buildTree("basil1", make_vec3(1.8, 1.4, 0));
    weberpenntree.buildTree("mint3", make_vec3(1.8, 1.3, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(1.8, 1.2, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(1.8, 1, 0));
    weberpenntree.buildTree("mint3", make_vec3(1.8, 0.8, 0));
    weberpenntree.buildTree("mint1", make_vec3(1.8, 0.6, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(1.8, 0.5, 0));
    weberpenntree.buildTree("mint1", make_vec3(1.8, 0.4, 0));
    weberpenntree.buildTree("basil1", make_vec3(1.8, 0.3, 0));
    weberpenntree.buildTree("basil2", make_vec3(1.8, 0.2, 0));
    weberpenntree.buildTree("mint2", make_vec3(1.7, 0.8, 0));

    weberpenntree.buildTree("bokchoy2", make_vec3(1.5, 0.8, 0));
    uint ID_bokchoy7 = weberpenntree.buildTree("bokchoy4", make_vec3(1.5, 0.5, 0));
    weberpenntree.buildTree("mint2", make_vec3(1.5, 0.5, 0));
    uint ID_bokchoy4 = weberpenntree.buildTree("bokchoy1", make_vec3(1.5, 0, 0));
    weberpenntree.buildTree("basil1", make_vec3(1.5, 0.2, 0));
    weberpenntree.buildTree("basil2", make_vec3(1.4, 1.8, 0));
    weberpenntree.buildTree("bokchoy2", make_vec3(1.4, 1.6, 0));
    weberpenntree.buildTree("basil3", make_vec3(1.4, 1.4, 0));
    weberpenntree.buildTree("mint3", make_vec3(1.4, 1.2, 0));
    weberpenntree.buildTree("basil3", make_vec3(1.4, 0.8, 0));
    weberpenntree.buildTree("basil2", make_vec3(1.4, 0, 0));
    weberpenntree.buildTree("bokchoy2", make_vec3(1.3, 0.5, 0));
    weberpenntree.buildTree("mint2", make_vec3(1.3, 0.5, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(1.2, 1.8, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(1.2, 1.3, 0));
    weberpenntree.buildTree("bokchoy4", make_vec3(1.2, 1, 0));

    weberpenntree.buildTree("mint3", make_vec3(1, 1.9, 0));
    weberpenntree.buildTree("mint1", make_vec3(1, 1.8, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(1, 1.6, 0));
    weberpenntree.buildTree("basil1", make_vec3(1, 1.4, 0));
    weberpenntree.buildTree("mint5", make_vec3(1, 1.0, 0));
    uint ID_mint = weberpenntree.buildTree("mint4", make_vec3(1, 0.5, 0));
    weberpenntree.buildTree("basil3", make_vec3(1, 0.2, 0));
    weberpenntree.buildTree("mint1", make_vec3(1, 0.2, 0));
    weberpenntree.buildTree("mint2", make_vec3(1, 0, 0));
    weberpenntree.buildTree("mint3", make_vec3(0.8, 0.1, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(0.7, 1.8, 0));
    weberpenntree.buildTree("mint1", make_vec3(0.75, 1.2, 0));
    weberpenntree.buildTree("basil1", make_vec3(0.7, 1.8, 0));
    weberpenntree.buildTree("mint3", make_vec3(0.7, 0.9, 0));
    weberpenntree.buildTree("basil3", make_vec3(0.7, 0.7, 0));
    weberpenntree.buildTree("mint2", make_vec3(0.7, 0.4, 0));
    weberpenntree.buildTree("basil2", make_vec3(0.7, 0.2, 0));

    weberpenntree.buildTree("bokchoy5", make_vec3(0.5, 2, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(0.5, 1.5, 0));
    weberpenntree.buildTree("bokchoy4", make_vec3(0.5, 1.2, 0));
    weberpenntree.buildTree("mint3", make_vec3(0.5, 1.0, 0));
    weberpenntree.buildTree("mint2", make_vec3(0.5, 0.6, 0));
    weberpenntree.buildTree("bokchoy2", make_vec3(0.5, 0.1, 0));
    weberpenntree.buildTree("mint1", make_vec3(0.4, 0.7, 0));
    weberpenntree.buildTree("mint3", make_vec3(0.4, 1.8, 0));
    weberpenntree.buildTree("basil3", make_vec3(0.3, 1.5, 0));
    weberpenntree.buildTree("mint3", make_vec3(0.25, 1.3, 0));
    weberpenntree.buildTree("mint1", make_vec3(0.2, 2, 0));
    weberpenntree.buildTree("mint2", make_vec3(0.2, 1.7, 0));
    weberpenntree.buildTree("basil1", make_vec3(0.2, 0.7, 0));
    weberpenntree.buildTree("mint2", make_vec3(0.2, 0.4, 0));
    uint ID_basil = weberpenntree.buildTree("basil4", make_vec3(0.2, 0.2, 0));
    weberpenntree.buildTree("mint2", make_vec3(0.1, 1.9, 0));
    weberpenntree.buildTree("bokchoy2", make_vec3(0, 2, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(0, 1.8, 0));
    weberpenntree.buildTree("bokchoy4", make_vec3(0, 1.5, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(0, 1.2, 0));
    weberpenntree.buildTree("bokchoy3", make_vec3(0, 1, 0));
    weberpenntree.buildTree("basil1", make_vec3(0, 1, 0));
    weberpenntree.buildTree("basil3", make_vec3(0, 0.75, 0));
    uint ID_bokchoy13 = weberpenntree.buildTree("bokchoy5", make_vec3(0, 0, 0));

    weberpenntree.buildTree("dill4", make_vec3(1.1, 1.5, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.9, 1.7, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.7, 1.6, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.7, 1.4, 0));
    weberpenntree.buildTree("dill4", make_vec3(0.9, 1.4, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.85, 1.6, 0));
    weberpenntree.buildTree("dill4", make_vec3(0.9, 1.3, 0));
    weberpenntree.buildTree("dill5", make_vec3(1.5, 0.3, 0));
    weberpenntree.buildTree("dill5", make_vec3(1.6, 1, 0));
    weberpenntree.buildTree("dill3", make_vec3(1.5, 1, 0));
    weberpenntree.buildTree("dill5", make_vec3(1.6, 1.2, 0));
    weberpenntree.buildTree("dill4", make_vec3(1.35, 0.3, 0));
    weberpenntree.buildTree("dill5", make_vec3(1.35, 0.1, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.5, 0.3, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.5, 0.4, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.3, 0.9, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.3, 1.1, 0));
    weberpenntree.buildTree("dill5", make_vec3(1, 0.8, 0));
    weberpenntree.buildTree("basil1", make_vec3(1.2, 0.7, 0));
    weberpenntree.buildTree("bokchoy1", make_vec3(1.1, 0.8, 0));
    weberpenntree.buildTree("dill5", make_vec3(0.9, 0.7, 0));

    // std::cout << std::to_string(ID_basil) << std::endl;

    // std::vector<uint> all_IDs = weberpenntree.getAllUUIDs(ID_basil);
    // std::cout << "After getting all IDs" << std::endl;

    vis.buildContextGeometry(&context); //add all geometry in the context to the visualizer
    vis.setCameraPosition(make_vec3(1, 1.1, 2.3), make_vec3(1, 1, 0));
    // vis.setCameraPosition(make_vec3(-0.5, -0.5, 1.2), make_vec3(1, 1, 0));
    vis.setCameraFieldOfView(75);
    vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
    vis.setLightDirection(make_vec3(1, 1, 5));

    // std::cout << "After building geometry" << std::endl;
    vis.plotUpdate(); //update the graphics window and move on
    // std::cout << "After plot update" << std::endl;
    struct timespec ts = {0, 100000000L};
    nanosleep(&ts, NULL);
    // std::string dir2 = "../videos/";
    // std::string name = "trees";
    // std::string ext = ".jpeg";
    // std::string file = dir2 + name + id + ext;
    // std::cout << file << std::endl;
    vis.printWindow("../garden.jpg"); //print window to JPEG file
    // std::cout << std::to_string(all_IDs.size()) << std::endl;
    vis.closeWindow();
    // vis.plotInteractive(); //open an interactive graphics window
}
