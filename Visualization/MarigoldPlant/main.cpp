#include "Context.h"
#include "Visualizer.h"

using namespace helios;

//this function builds a "prototype" leaf that has length and width of unity and has the base of the leaf located at the origin. The leaf is made up of a mesh of triangles, which allows us to give it some curvature. The curvature is given by an exponential function.
std::vector<uint> addBasilLeaf( helios::Context* context ){

  int2 submesh_res(10,6);   //resolution of the submesh of triangles in x- and y-directions
  //exponential function for curvature is z = a0*(1-exp(e0*M)), where M is the distance from the base of the leaf
  float a0 = 0.0025;         
  float e0 = 5.f;
  
  std::vector<uint> UUIDs;

  float dx = 1.f/float(submesh_res.x);
  float dy = 1.f/float(submesh_res.y);

  float x,y,z,mag;
  for( int j=0; j<submesh_res.y; j++ ){
    for( int i=0; i<submesh_res.x; i++ ){

      //define the four vertices of each triangle gridcell
      
      x = i*dx;
      y = -0.5+j*dy;
      
      mag = sqrt( x*x + 2*y*y );
      z = a0*(1.f-exp(e0*mag));
      vec3 v0( x, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*y*y );
      z = a0*(1.f-exp(e0*mag));
      vec3 v1( x+dx, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*(y+dy)*(y+dy) );
      z = a0*(1.f-exp(e0*mag));
      vec3 v2( x+dx, y+dy, z );

      mag = sqrt( x*x + 2*(y+dy)*(y+dy) );
      z = a0*(1.f-exp(e0*mag));
      vec3 v3( x, y+dy, z );

      //define the four (u,v) texture coordinates for the triangle gridcell
      vec2 uv0( x, j*dy);
      vec2 uv1( x+dx, j*dy );
      vec2 uv2( x+dx, (j+1)*dy );
      vec2 uv3( x, (j+1)*dy );

      //adding the triangles to the context
      UUIDs.push_back( context->addTriangle( v0, v1, v2, "../textures/Marigold1.png", uv0, uv1, uv2 ) );
      UUIDs.push_back( context->addTriangle( v0, v2, v3, "../textures/Marigold1.png", uv0, uv2, uv3 ) );

    }
  }

  //rotate about y-axis by -0.4 radians to make the leaf roughly horizontal
  context->rotatePrimitive( UUIDs, -0.4, "y" );
  
  return UUIDs;

}

std::vector<uint> addPlanterBox( helios::vec3 base, helios::vec3 size , helios::vec2 sprayer_position, bool sprayer_on, helios::Context* context ){

  const char* texture = "../textures/MetalTexture.jpg";   //texture for box
  float lip_width = 0.1;      //width of the box lip
  vec3 post_size(0.05,0.05,3);   //size of the corner posts
  float h=0.5; //height of horizontal supports

  std::vector<uint> UUIDs, U;
  
  //-- outer sides --//

  UUIDs.push_back( context->addPatch( base+make_vec3(0, 0.5*size.y+lip_width, 0.5*size.z), make_vec2(size.x+2*lip_width,size.z), make_SphericalCoord(0.5*M_PI,0), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(0, -0.5*size.y-lip_width, 0.5*size.z), make_vec2(size.x+2*lip_width,size.z), make_SphericalCoord(-0.5*M_PI,0), texture) );

  UUIDs.push_back( context->addPatch( base+make_vec3(0.5*size.x+lip_width, 0, 0.5*size.z), make_vec2(size.y+2*lip_width,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(-0.5*size.x-lip_width, 0, 0.5*size.z), make_vec2(size.y+2*lip_width,size.z), make_SphericalCoord(0.5*M_PI,-0.5*M_PI), texture) );

  //-- lip --//

  UUIDs.push_back( context->addPatch( base+make_vec3(0, 0.5*size.y+0.5*lip_width, size.z), make_vec2(size.x+2*lip_width,lip_width), make_SphericalCoord(0,0), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(0, -0.5*size.y-0.5*lip_width, size.z), make_vec2(size.x+2*lip_width,lip_width), make_SphericalCoord(0,0), texture) );

  UUIDs.push_back( context->addPatch( base+make_vec3(0.5*size.x+0.5*lip_width, 0, size.z), make_vec2(lip_width,size.y), make_SphericalCoord(0,0), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(-0.5*size.x-0.5*lip_width, 0, size.z), make_vec2(lip_width,size.y), make_SphericalCoord(0,0), texture) );

  //-- inner sides --//
  UUIDs.push_back( context->addPatch( base+make_vec3(0, 0.5*size.y, 0.5*size.z), make_vec2(size.x,size.z), make_SphericalCoord(-0.5*M_PI,0), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(0, -0.5*size.y, 0.5*size.z), make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), texture) );

  UUIDs.push_back( context->addPatch( base+make_vec3(0.5*size.x, 0, 0.5*size.z), make_vec2(size.y,size.z), make_SphericalCoord(-0.5*M_PI,0.5*M_PI), texture) );
  UUIDs.push_back( context->addPatch( base+make_vec3(-0.5*size.x, 0, 0.5*size.z), make_vec2(size.y,size.z), make_SphericalCoord(-0.5*M_PI,-0.5*M_PI), texture) );

  //-- vertical posts --//
  U = context->addBox( base+make_vec3( -0.5*size.x-0.5*post_size.x, -0.5*size.y-0.5*post_size.y, 0.8*size.z+0.5*post_size.z), post_size, make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( 0.5*size.x+0.5*post_size.x, -0.5*size.y-0.5*post_size.y, 0.8*size.z+0.5*post_size.z), post_size, make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( 0.5*size.x+0.5*post_size.x, 0.5*size.y+0.5*post_size.y, 0.8*size.z+0.5*post_size.z), post_size, make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( -0.5*size.x-0.5*post_size.x, 0.5*size.y+0.5*post_size.y, 0.8*size.z+0.5*post_size.z), post_size, make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );

  //-- horizontal supports --//
  U = context->addBox( base+make_vec3( -0.5*size.x-post_size.x, 0, 0.8*size.z+post_size.z-0.5*h), make_vec3(post_size.x, size.y, h), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( 0.5*size.x+post_size.x, 0, 0.8*size.z+post_size.z-0.5*h), make_vec3(post_size.x, size.y, h), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( 0, -0.5*size.y-post_size.y, 0.8*size.z+post_size.z-0.5*h), make_vec3( size.x, post_size.y, 0.5), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  U = context->addBox( base+make_vec3( 0, 0.5*size.y+post_size.y, 0.8*size.z+post_size.z-0.5*h), make_vec3( size.x, post_size.y, 0.5), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );

  //-- sprayer --//
  U = context->addBox( base+make_vec3( -0.5*size.x+sprayer_position.x*size.x, 0, 0.8*size.z+post_size.z-0.5*h), make_vec3(8*post_size.x, size.y, h), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );

  U = context->addBox( base+make_vec3( -0.5*size.x+sprayer_position.x*size.x, -0.5*size.y+sprayer_position.y*size.y, 0.8*size.z+post_size.z-0.5*post_size.y-0.1 ), make_vec3(0.2,0.2,0.2), make_int3(1,1,1), texture );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  
  std::vector<vec3> nodes;
  std::vector<float> radius;
  std::vector<RGBcolor> color;
  nodes.push_back( base+make_vec3( -0.5*size.x+sprayer_position.x*size.x, -0.5*size.y+sprayer_position.y*size.y, 0.8*size.z+post_size.z-0.2 ) );
  nodes.push_back( base+make_vec3( -0.5*size.x+sprayer_position.x*size.x, -0.5*size.y+sprayer_position.y*size.y, 0.8*size.z+post_size.z-1.f ) );
  radius.push_back( 0.05 );
  radius.push_back( 0.05 );
  color.push_back( RGB::blue );
  color.push_back( RGB::blue );
  U = context->addTube( 25, nodes, radius, color );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );

  U = context->addDisk( 25, nodes.back(), make_vec2( 0.15, 0.15), make_SphericalCoord(0,0), RGB::blue );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );
  
  if( sprayer_on ){

    int Ndrops = 5000;
    for( int i=0; i<Ndrops; i++ ){
    
      float rtheta = 2*M_PI*context->randu();
      float rz = context->randu();
      
      float z1 = base.z+0.8*size.z+post_size.z-1.f;
      float z0 = base.z+0.8*size.z;
      float z = z0+(z1-z0)*rz;
      
      float radius = 1.8*(z1-z)/(z1-z0);
      float r = radius*context->randu();
      
      float x = base.x-0.5*size.x+sprayer_position.x*size.x+ r*cos(rtheta);
      float y = base.y-0.5*size.y+sprayer_position.y*size.y+ r*sin(rtheta);
    
      context->addSphere( 6, make_vec3(x,y,z), 0.002, RGB::cyan );
    
    }
    
  }

  //-- soil --//

  UUIDs.push_back( context->addPatch( base+make_vec3(0,0,0.8*size.z), make_vec2(size.x,size.y), make_SphericalCoord(0,0), "plugins/visualizer/textures/dirt.jpg" ) );

  //-- text --//

  UUIDs.push_back( context->addPatch( base+make_vec3(0,0.5*size.y+0.101,0.5*size.z), make_vec2(0.5*size.x,0.25*size.z), make_SphericalCoord(-0.5*M_PI,M_PI), "../textures/AlphaGardenTexture.png" ) );
  
  return UUIDs;

}

std::vector<uint> addBasilPlant( helios::vec3 base, float height, helios::Context* context, int random_seed ){

  float azimuth_var = 1.f;       //range of variation for leaf azimuth
  float internode_var = 0.1;     //range of variation for internode spacing
  float height_var = 0.4;        //range of variation for plant height
  float leafsize_var = 0.05;     //range of variation for leaf size
  float leafangle_var = 1.0;     //range of variation for leaf inclination
  float leafangle_jitter = 0.05; //range of variation for random leaf angle jitter
  float position_var = 0.5;      //range of variation for position of the plant base
  //float stemangle_var = 0.5;
  
  float stem_radius = 0.005;     //radius of the main stem
  RGBcolor stem_color(0.5,0.4,0.3);  //color of the main stem

  float internode_spacing = 0.12;    //average internode spacing

  vec2 leaf_size(0.3,0.2);       //average leaf size in x- and y- directions
  
  std::vector<uint> UUIDs;

  //In order to have repeatable plants for every frame, we'll seed the random number generator with the same value every time, but different seeds for different plants
  std::minstd_rand0 generator;
  std::uniform_real_distribution<float> unif_distribution;
  generator.seed(random_seed);
  unif_distribution(generator);//not sure why, but I think has to be called once to get going

  //---------- MAIN STEM ----------- //

  float h_plant = height-0.5*unif_distribution(generator)*height_var; //random height
  //float h_plant = height;   //constant height

  base.x += -0.5*unif_distribution(generator)*position_var;
  base.y += -0.5*unif_distribution(generator)*position_var;
  
  std::vector<vec3> stem_nodes;
  std::vector<float> stem_rad;
  std::vector<RGBcolor> stem_c;

  float stem_theta = 0.f;
  float stem_azimuth = 0.f;
  
  for( int i=0; i<10; i++ ){
    
    stem_nodes.push_back( base+make_vec3( sin(stem_theta*i/9.f)*sin(stem_azimuth), sin(stem_theta*i/9.f)*cos(stem_azimuth), i*h_plant/9.f) );
    stem_rad.push_back( stem_radius*(12.f-i)/9.f );
    stem_c.push_back( stem_color );
    
  }

  std::vector<uint> U = context->addTube( 15, stem_nodes, stem_rad, stem_c );
  UUIDs.insert( UUIDs.begin(), U.begin(), U.end() );

  //---------- LEAVES ----------- //

  //when you have texture-mapped triangles in Helios, it is MUCH faster to copy a leaf and scale/translate/rotate it than creating a new leaf from scratch. so we'll make one "prototye" that we can copy over and over then scale/translate/rotate it. 
  std::vector<uint> leaf_prototype = addBasilLeaf( context );

  //number of nodes (leaf pairs) along the stem
  int node_count = floor( h_plant/internode_spacing );

  float azimuth_plant = 2.f*M_PI*unif_distribution(generator);
  
  std::vector<uint> UUID_leaf1;
  for( int i=0; i<node_count+1; i++ ){

    UUID_leaf1 = context->copyPrimitive( leaf_prototype );

    if( i<node_count ){
      context->scalePrimitive( UUID_leaf1, make_vec3(leaf_size.x,leaf_size.y,leaf_size.x)*(1.f-0.5*unif_distribution(generator)*leafsize_var) );

    }else{
      float mscale = (h_plant-node_count*internode_spacing)/internode_spacing;
      context->scalePrimitive( UUID_leaf1, mscale*make_vec3(leaf_size.x,leaf_size.y,leaf_size.x)*(1.f-0.5*unif_distribution(generator)*leafsize_var) );

    }

    float jitterx = -0.5*context->randu()*leafangle_jitter;
    float jittery = -0.5*context->randu()*leafangle_jitter;
    context->rotatePrimitive( UUID_leaf1, jitterx-0.5*unif_distribution(generator)*leafangle_var, "x" );
    context->rotatePrimitive( UUID_leaf1, jittery-0.5*unif_distribution(generator)*leafangle_var, "y" );

    context->rotatePrimitive( UUID_leaf1, azimuth_plant+(0.76389*M_PI*i)-0.5*unif_distribution(generator)*azimuth_var, "z" );

    float z_rand = -0.5*unif_distribution(generator)*internode_var;
    if( i<node_count ){
      context->translatePrimitive( UUID_leaf1, base+make_vec3(0,0,(i+1)*internode_spacing+z_rand) );
    }else{
      context->translatePrimitive( UUID_leaf1, base+make_vec3(0,0,h_plant+z_rand) );
    }

  }
  
  //clean up afterword
  context->deletePrimitive( leaf_prototype );

  return UUIDs;

}

int main( void ){

  float spacing = 0.75;     //average spacing between plants
  vec3 boxsize(10,5,2);     //size of the planter box in x-, y- and z-directons

  float dx_tile = 5;        //width of floor tiles
  int Ntile = 6;            //number of floor tiles

  int Nt = 50;              //number of movie frames

  for( int t=0; t<Nt; t++ ){
  
    Context context;

    //add the floor (concrete)
    for( int j=0; j<Ntile; j++ ){
      for( int i=0; i<Ntile; i++ ){

	context.addPatch( make_vec3(-0.5*Ntile*dx_tile+i*dx_tile,-0.5*Ntile*dx_tile+j*dx_tile,0), make_vec2(dx_tile,dx_tile), make_SphericalCoord(0,0), "../textures/ConcreteTexture.jpg" );

      }
    }

    //define the position of the sprayer
//    vec2 sprayer_pos( fmin(0.6,0.1+0.5*float(t+1)/float(Nt*0.7)), 0.5); //moving sprayer
    vec2 sprayer_pos( 0.05, 0.5);  //stationary sprayer

    bool sprayer_on = false;
    if( t>0.7*Nt ){
      sprayer_on = true;  //this turns the sprayer on
    }

    //build the planter box
    addPlanterBox( make_vec3(0,0,0), boxsize, sprayer_pos, sprayer_on,  &context );

    //build the plants
    int2 Np( floor(boxsize.x/spacing), floor(boxsize.y/spacing) );

    float height_average = float(t)/float(Nt-1);  //growing plants
//    float height_average = 1.f;  //"fully grown" plants
    
    for( int j=0; j<Np.y; j++ ){
      for( int i=0; i<Np.x; i++ ){
	
	vec3 position( -0.5*Np.x*spacing+(i+0.5)*spacing, -0.5*Np.y*spacing+(j+0.5)*spacing,0.85*boxsize.z );
	
	addBasilPlant( position, height_average, &context, j*Np.x+i );
	
      }
    }

    //visualizer stuff
    
    Visualizer vis(1200);

    vis.setBackgroundColor( RGB::white );

    vis.buildContextGeometry(&context);
   
    vis.setCameraPosition( make_SphericalCoord(7.3,0.35,0), make_vec3(0,0,2.0) ); //view for growing plants
//    vis.setCameraPosition( make_SphericalCoord(9,0.15,0), make_vec3(0,0,2.0) ); //view for moving sprayer
    
    vis.setLightDirection( make_vec3(0.01,0.01,1) );
    vis.setLightingModel( Visualizer::LIGHTING_PHONG_SHADOWED );
 
    vis.plotUpdate();

    wait(1);

    char filename[50];
    sprintf(filename,"../frames/marigold_%03d.jpeg",t);
    vis.printWindow(filename);
    

  }
  
}
