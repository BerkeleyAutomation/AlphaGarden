#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"

using namespace helios;

int main( void ){
    Context context; //declare the context

    //create an instance of the WeberPennTree class, which we will call "weberpenntree"
    WeberPennTree weberpenntree( &context );
    //Create an almond tree at the point (0,0,0)
    // uint ID_almond = weberpenntree.buildTree( "Almond", make_vec3(0,10,0) );
    //Retrieve UUIDs for context primitives making up the almond tree's leaves
    // std::vector<uint> leafUUIDs_almond = weberpenntree.getLeafUUIDs( ID_almond );
    // uint ID_avocado = weberpenntree.buildTree( "Avocado", make_vec3(10,0,0) );

    // weberpenntree.loadXML("../tools/basil1.xml");
    // weberpenntree.loadXML("../tools/basil2.xml");
    // weberpenntree.loadXML("../tools/basil6.xml");

    for(int i=80;i<100;i++) {
        Visualizer vis(1200); //Opens a graphics window of width 1200 pixels with default aspect ratio
        // vis.addSkyDomeByCenter(100, make_vec3(0, 0, 0), 20, "plugins/visualizer/textures/SkyDome_clouds.jpg", 0);

        std::string label = "basil";
        std::string dir = "../../tools/";
        std::string xml_ext = ".xml";
        std::string id = std::to_string(i);
        weberpenntree.loadXML((dir + label + id + xml_ext).c_str());

        uint ID_basil = weberpenntree.buildTree((label + id).c_str(), make_vec3(0, 0, 0));
        // std::cout << std::to_string(ID_basil) << std::endl;

        std::vector<uint> all_IDs = weberpenntree.getAllUUIDs(ID_basil);
        // std::cout << "After getting all IDs" << std::endl;

        vis.buildContextGeometry(&context); //add all geometry in the context to the visualizer
        vis.setCameraPosition(make_vec3(0.1, 0.1, 3.1), make_vec3(0, 0, 0));
        vis.setCameraFieldOfView(60);
        vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
        vis.setLightDirection(make_vec3(1, 1, 1));

        std::cout << "After building geometry" << std::endl;
        vis.plotUpdate();                   //update the graphics window and move on
        std::cout << "After plot update" << std::endl;
        struct timespec ts = {0, 100000000L};
        nanosleep(&ts, NULL);
        std::string dir2 = "../videos/";
        std::string name = "trees";
        std::string ext = ".jpeg";
        std::string file = dir2 + name + id + ext;
        // std::cout << file << std::endl;
        vis.printWindow(file.c_str()); //print window to JPEG file
        // std::cout << std::to_string(all_IDs.size()) << std::endl;
        vis.closeWindow();
        // vis.plotInteractive(); //open an interactive graphics window

        vis.clearGeometry();
        context.deletePrimitive(context.getAllUUIDs());
    }
}
