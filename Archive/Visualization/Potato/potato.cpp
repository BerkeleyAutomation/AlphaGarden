#include "potato.h"

using namespace helios;

std::vector<uint> potato( const vec3 center, const float scale, Context* context ){

  float height = 1.f;
  
  float h_std = 0.00;      //standard deviation of plant height

  float r_stem = 0.02;     //radius of stem base

  float pct_base = 0.15;    //percentage of plant height before branches start

  float stem_angle = 0.05; //standard deviation of stem tilt/curve angle

  float l_leaf_max = 0.25;  //maximum (distal) length of leaves

  float l_branch = 0.3;

  int Nstems = 30;

  std::vector<uint> UUIDs;

  // stem
  std::vector<vec3> nodes;
  std::vector<float> radius;

  uint Nstemsegs = 10;

  float phi_tilt = context->randu()*2.f*M_PI;
  float h_plant = (height + context->randn()*h_std)*scale;
  
  for( int i=0; i<Nstemsegs; i++ ){

    float theta_tilt = stem_angle*float(i)/float(Nstemsegs-1);
    
    vec3 p = center + make_vec3( sin(theta_tilt)*cos(phi_tilt), sin(theta_tilt)*sin(phi_tilt), h_plant*float(i)/float(Nstemsegs-1) );

    nodes.push_back(p);

    radius.push_back( scale*r_stem*(1.f-float(i)/float(Nstemsegs-1)) );
    
  }

  std::vector<uint> UUIDs_tube = context->addTube( 8, nodes, radius );

  UUIDs.insert( UUIDs.begin(), UUIDs_tube.begin(), UUIDs_tube.end() );

  // branches

  std::vector<float> phi_branch;
  phi_branch.resize(Nstems);
  phi_branch.at(0) = 0.f;
  for( int i=1; i<Nstems; i++ ){
    //phi_branch.push_back(context->randu()*2.f*M_PI);
    phi_branch.at(i) = phi_branch.at(i-1)+55.f*M_PI/180.f;
  }
  
  for( int i=0; i<Nstems-1; i++ ){

    nodes.resize(0);
    radius.resize(0);

    float z = h_plant*pct_base+h_plant*(1.f-pct_base)*float(i)/float(Nstems-1);
    float theta_tilt = stem_angle*z/h_plant;
    vec3 p = center + make_vec3( sin(theta_tilt)*cos(phi_tilt), sin(theta_tilt)*sin(phi_tilt), z );

    
    for( int j=0; j<Nstemsegs; j++ ){

      float f = float(j)/float(Nstemsegs-1);
      
      vec3 p_branch = p+make_vec3( sin(phi_branch.at(i))*l_branch*f*scale, cos(phi_branch.at(i))*l_branch*f*scale, 0.6/(0.35+f)*l_branch*f*scale );

      nodes.push_back(p_branch);

      radius.push_back( scale*0.5*r_stem*(1.f-float(i)/float(Nstems-1))*(1.f-f) );

    }

    //std::vector<uint> UUIDs_tube = context->addTube( 8, nodes, radius );
    //UUIDs.insert( UUIDs.begin(), UUIDs_tube.begin(), UUIDs_tube.end() );

  }

  //leaves

  for( int i=0; i<Nstems-1; i++ ){

    float z = h_plant*pct_base+h_plant*(1.f-pct_base)*float(i)/float(Nstems-1);
    float theta_tilt = stem_angle;
    vec3 p = center + make_vec3( sin(theta_tilt)*cos(phi_tilt), sin(theta_tilt)*sin(phi_tilt), z );

    float theta_branch = 0.f;
    vec3 p_leaf;
    float l_leaf;
    float f;
    uint UUID;

    f = 1;
    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.3*l_leaf_max*scale,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( -theta_branch+0.3+context->randn(0,0.4), "y" );
    context->getPrimitivePointer(UUID)->rotate( 0.5*M_PI-phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);


    l_leaf = 0.6*l_leaf_max*scale;
    f = 0.9;

    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( -theta_branch+context->randn(0,0.4), "x" );
    context->getPrimitivePointer(UUID)->rotate( -phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);

    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( theta_branch+context->randn(0,0.4), "x" );
    context->getPrimitivePointer(UUID)->rotate( M_PI-phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);


    l_leaf = 0.4*l_leaf_max*scale;
    f = 0.6;
    
    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( -theta_branch+context->randn(0,0.4), "x" );
    context->getPrimitivePointer(UUID)->rotate( -phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);

    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( theta_branch+context->randn(0,0.4), "x" );
    context->getPrimitivePointer(UUID)->rotate( M_PI-phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);

    l_leaf = 0.25*l_leaf_max*scale;
    f = 0.4;
    
    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( -theta_branch+context->randn(0,0.3), "x" );
    context->getPrimitivePointer(UUID)->rotate( -phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);

    p_leaf = p + make_vec3( sin(phi_branch.at(i)), cos(phi_branch.at(i)), 0.6/(0.35+f) )*l_branch*f*scale;
    UUID = context->addPatch( make_vec3(0.35*l_leaf,0,0), make_vec2(scale*l_leaf_max,scale*0.5*l_leaf_max), make_SphericalCoord(0,0), "plugins/visualizer/textures/AspenLeaf.png" );
    context->getPrimitivePointer(UUID)->rotate( theta_branch+context->randn(0,0.3), "x" );
    context->getPrimitivePointer(UUID)->rotate( M_PI-phi_branch.at(i), "z" );
    context->getPrimitivePointer(UUID)->translate( p_leaf );

    UUIDs.push_back(UUID);
    
  }

  for( int p=UUIDs.size()-1; p>=0; p-- ){
    float area = context->getPrimitivePointer( UUIDs.at(p) )->getArea();
    if( area<1e-12 ){
      context->deletePrimitive(UUIDs.at(p));
      UUIDs.erase(UUIDs.begin()+p);
    }
  }

  return UUIDs;

}
