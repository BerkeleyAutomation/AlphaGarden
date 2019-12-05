#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"

using namespace helios;

int main( void ){
    Context context; //declare the context

    //create an instance of the WeberPennTree class, which we will call "weberpenntree"
    WeberPennTree weberpenntree( &context );
    //Create an almond tree at the point (0,0,0)
    // uint ID_almond = weberpenntree.buildTree( "Almond", make_vec3(0,0,0) );
    //Create an orange tree at the point (10,0,0)
    // uint ID_orange = weberpenntree.buildTree( "Orange", make_vec3(10,0,0) );
    //Retrieve UUIDs for context primitives making up the almond tree's leaves
    // std::vector<uint> leafUUIDs_almond = weberpenntree.getLeafUUIDs( ID_almond );
    weberpenntree.loadXML("../../tools/hardcoded/basilsmall.xml");
    weberpenntree.buildTree("basilsmall", make_vec3(0,0,0));

    Visualizer vis(1200); //creates a display window 800 pixels wide
    vis.buildContextGeometry( &context ); //add all geometry in the context to the visualizer
    vis.plotUpdate(); //update the graphics window and move on
    // vis.printWindow( "trees.jpeg" ); //print window to JPEG file
    vis.plotInteractive(); //open an interactive graphics window

}
