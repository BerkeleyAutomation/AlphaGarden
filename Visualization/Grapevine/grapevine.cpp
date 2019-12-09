#include "grapevine.h"

#include <random>
#include <chrono>

using namespace helios;
using namespace std;

float sampleLeafAngle( std::default_random_engine generator ){

  //define leaf angle PDF (from data)
  std::vector<float> leafAngleDist;
  leafAngleDist.push_back( 2 );
  leafAngleDist.push_back( 1 );
  leafAngleDist.push_back( 1 );
  leafAngleDist.push_back( 3 );
  leafAngleDist.push_back( 4 );
  leafAngleDist.push_back( 4 );
  leafAngleDist.push_back( 13 );
  leafAngleDist.push_back( 10 );
  leafAngleDist.push_back( 15 );
  leafAngleDist.push_back( 27 );

  float dTheta = 0.5f*M_PI/float(leafAngleDist.size());

  //make sure PDF is properly normalized
  float norm = 0;
  for( int i=0; i<leafAngleDist.size(); i++ ){
    norm += leafAngleDist.at(i)*dTheta;
  }
  for( int i=0; i<leafAngleDist.size(); i++ ){
    leafAngleDist.at(i) /= norm;
  }
  norm = 0;
  for( int i=0; i<leafAngleDist.size(); i++ ){
    norm += leafAngleDist.at(i)*dTheta;
  }
  assert( fabs(norm-1)<0.001 );
  
  //calculate the leaf angle CDF
  std::vector<float> leafAngleCDF;
  leafAngleCDF.resize( leafAngleDist.size() );
  float tsum = 0;
  for( int i=0; i<leafAngleDist.size(); i++ ){
    tsum += leafAngleDist.at(i)*dTheta;
    leafAngleCDF.at(i) = tsum;
  }

  assert( fabs(tsum-1.f)<0.001 );
  
  //draw from leaf angle PDF
  std::uniform_real_distribution<float> unif_distribution;
  float rt = unif_distribution(generator);

  float theta = -1;
  for( int i=0; i<leafAngleCDF.size(); i++ ){
    if( rt<leafAngleCDF.at(i) ){
      theta = (i+unif_distribution(generator))*dTheta;
      break;
    }
  }

  assert( theta!=-1 );
  
  return theta;
  
}

vector<vector<uint> > grapevine( vec3 origin, float width, float height, float orientation, Context* context ){

  vector<vector<uint> > UUIDs;
  vector<uint> U;

  float trunk_height = 0.35*height;
  float trunk_radius = 0.04*trunk_height;
  float cane_radius = 0.15*trunk_radius;
  float shoot_radius = 0.5*cane_radius;
  RGBcolor wood_color = make_RGBcolor(0.36,0.24,0.21);
  RGBcolor leaf_color = make_RGBcolor(0.35,0.54,0.12);

  uint shoots_per_cane = 14;
  float mean_shoot_angle = 20;

  uint leaves_per_shoot = 22;
  float leaf_size = 0.17;

  std::default_random_engine generator;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);

  std::uniform_real_distribution<float> unif_distribution;

  float trunk_std = 0.025*trunk_height;
  
  //------ trunks -------//

  std::vector<float> rad_main;
  rad_main.push_back(trunk_radius);
  rad_main.push_back(0.9f*trunk_radius+unif_distribution(generator)*trunk_std);
  rad_main.push_back(0.8f*trunk_radius+unif_distribution(generator)*trunk_std);
  rad_main.push_back(trunk_radius);
  std::vector<vec3> pos_main;
  pos_main.push_back(make_vec3(0.,0.,0.0));
  pos_main.push_back(make_vec3(0+unif_distribution(generator)*trunk_std,0.+unif_distribution(generator)*trunk_std,0.4f*trunk_height));
  pos_main.push_back(make_vec3(0.+unif_distribution(generator)*trunk_std,0.+unif_distribution(generator)*trunk_std,0.6f*trunk_height));
  pos_main.push_back(make_vec3(0.,0.,0.7f*trunk_height));

  std::vector<RGBcolor> trunk_color;
  for( uint i=0; i<rad_main.size(); i++ ){
    trunk_color.push_back(wood_color);
    pos_main.at(i) = pos_main.at(i) + origin;
  }
      
  //U = context->addTube(30,pos_main,rad_main,trunk_color);
  U = context->addTube(30,pos_main,rad_main, "plugins/visualizer/textures/wood2.jpg" );
  UUIDs.push_back(U);
  context->setPrimitiveData(U,"element_type","trunk");

  //------ canes -------//

  float theta = orientation;
 
  //West Cane
  std::vector<float> rad_canew;
  rad_canew.push_back(3.5f*cane_radius);
  rad_canew.push_back(3.3f*cane_radius);
  rad_canew.push_back(3.0f*cane_radius);
  rad_canew.push_back(2.5f*cane_radius);
  rad_canew.push_back(2.f*cane_radius);
  rad_canew.push_back(1.f*cane_radius);
  rad_canew.push_back(0.75f*cane_radius);
  std::vector<vec3> pos_canew;
  // pos_canew.push_back(make_vec3(0.+0,0.,0.7f*trunk_height));
  // pos_canew.push_back(make_vec3(0.+0.075f,0.,0.8f*trunk_height));
  // pos_canew.push_back(make_vec3(0.+0.15f,0.,0.85f*trunk_height));
  // pos_canew.push_back(make_vec3(0.+0.45f+unif_distribution(generator)*trunk_std,0.-0.04f+unif_distribution(generator)*trunk_std,0.9f*trunk_height+unif_distribution(generator)*trunk_std));
  // pos_canew.push_back(make_vec3(0.+0.6f,0.-0.04f+unif_distribution(generator)*trunk_std,0.95f*trunk_height+unif_distribution(generator)*trunk_std));
  // pos_canew.push_back(make_vec3(0.+0.85f,0.-0.04f,1.f*trunk_height));
  // pos_canew.push_back(make_vec3(0.+0.5f*width,0.-0.04f,1.f*trunk_height));
  pos_canew.push_back(make_vec3(0,0,0.7f*trunk_height));
  pos_canew.push_back(make_vec3(0.075*cos(theta),0.075*sin(theta),0.8f*trunk_height));
  pos_canew.push_back(make_vec3(0.15f*cos(theta),0.15f*sin(theta),0.85f*trunk_height));
  float r = unif_distribution(generator)*trunk_std;
  pos_canew.push_back(make_vec3((0.45f+r)*cos(theta),(0.45f+r)*sin(theta)+(-0.04f+unif_distribution(generator)*trunk_std),0.9f*trunk_height+unif_distribution(generator)*trunk_std));
  pos_canew.push_back(make_vec3(0.6f*cos(theta),0.6f*sin(theta)+(-0.04f+unif_distribution(generator)*trunk_std)*sin(theta),0.95f*trunk_height+unif_distribution(generator)*trunk_std));
  pos_canew.push_back(make_vec3(0.85f*cos(theta),0.85f*sin(theta)-0.04f,1.f*trunk_height));
  pos_canew.push_back(make_vec3(0.5f*width*cos(theta),0.5f*width*sin(theta)-0.04f*sin(theta),1.f*trunk_height));
	
  std::vector<RGBcolor> cane_color;
  for( uint i=0; i<rad_canew.size(); i++ ){
    cane_color.push_back(wood_color);
  }

  std::vector<vec3> tmp;
  tmp.resize(pos_canew.size());
  for( uint i=0; i<pos_canew.size(); i++ ){
    tmp.at(i) = pos_canew.at(i) + origin;
  }

  U = context->addTube(10,tmp,rad_canew, "plugins/visualizer/textures/wood2.jpg");
  UUIDs.push_back(U);
  context->setPrimitiveData(U,"element_type","cane");

  //East Cane
  std::vector<float> rad_canee;
  rad_canee.push_back(3.5f*cane_radius);
  rad_canee.push_back(3.3f*cane_radius);
  rad_canee.push_back(3.0f*cane_radius);
  rad_canee.push_back(2.5f*cane_radius);
  rad_canee.push_back(2.f*cane_radius);
  rad_canee.push_back(1.f*cane_radius);
  rad_canee.push_back(0.75f*cane_radius);
  std::vector<vec3> pos_canee;
  //canes run along x-axis
  // pos_canee.push_back(make_vec3(0.-0,0.,0.7f*trunk_height));
  // pos_canee.push_back(make_vec3(0.-0.075f,0.,0.8f*trunk_height));
  // pos_canee.push_back(make_vec3(0.-0.15f,0.,0.85f*trunk_height));
  // pos_canee.push_back(make_vec3(0.-0.45f+unif_distribution(generator)*trunk_std,0.-0.04f+unif_distribution(generator)*trunk_std,0.9f*trunk_height+unif_distribution(generator)*trunk_std));
  // pos_canee.push_back(make_vec3(0.-0.6f,0.-0.04f+unif_distribution(generator)*trunk_std,0.95f*trunk_height+unif_distribution(generator)*trunk_std));
  // pos_canee.push_back(make_vec3(0.-0.85f,0.-0.04f,1.f*trunk_height));
  // pos_canee.push_back(make_vec3(0.-0.5f*width,0.-0.04f,1.f*trunk_height));
  pos_canee.push_back(make_vec3(0,0,0.7f*trunk_height));
  pos_canee.push_back(make_vec3(-0.075*cos(theta),-0.075*sin(theta),0.8f*trunk_height));
  pos_canee.push_back(make_vec3(-0.15f*cos(theta),-0.15f*sin(theta),0.85f*trunk_height));
  r = unif_distribution(generator)*trunk_std;
  pos_canee.push_back(make_vec3(-(0.45f+r)*cos(theta),-(0.45f+r)*sin(theta)+(-0.04f+unif_distribution(generator)*trunk_std),0.9f*trunk_height+unif_distribution(generator)*trunk_std));
  pos_canee.push_back(make_vec3(-0.6f*cos(theta),-0.6f*sin(theta)+(-0.04f+unif_distribution(generator)*trunk_std)*sin(theta),0.95f*trunk_height+unif_distribution(generator)*trunk_std));
  pos_canee.push_back(make_vec3(-0.85f*cos(theta),-0.85f*sin(theta)-0.04f,1.f*trunk_height));
  pos_canee.push_back(make_vec3(-0.5f*width*cos(theta),-0.5f*width*sin(theta)-0.04f*sin(theta),1.f*trunk_height));
  
  tmp.resize(pos_canee.size());
  for( uint i=0; i<pos_canee.size(); i++ ){
    tmp.at(i) = pos_canee.at(i) + origin;
  }
  
  U = context->addTube(10,tmp,rad_canee, "plugins/visualizer/textures/wood2.jpg");
  UUIDs.push_back(U);
  context->setPrimitiveData(U,"element_type","cane");

  //------- primary shoots ---------//

  std::vector<RGBcolor> ctable;
  ctable.push_back( make_RGBcolor( 0.6, 0, 0 ) );
  ctable.push_back( RGB::red );
  ctable.push_back( RGB::white);
  ctable.push_back( RGB::green );
  ctable.push_back( make_RGBcolor( 0, 0.6, 0 ) );
  
  std::vector<float> clocs;
  clocs.push_back( 0 );
  clocs.push_back( 0.25 );
  clocs.push_back( 0.5 );
  clocs.push_back( 0.7 );
  clocs.push_back( 1.0 );
  
  //West Cane

  for( uint c=0; c<2; c++ ){

    std::vector<float> rad_cane;
    std::vector<vec3> pos_cane;
    float sign;
    if( c==0 ){
      pos_cane = pos_canew;
      rad_cane = rad_canew;
      sign = 1;
    }else{
      pos_cane = pos_canee;
      rad_cane = rad_canee;
      sign = -1;
    }

    float dx = fabs(pos_cane.back().y-pos_cane.at(0).y)/(float(shoots_per_cane));
    
    for( int j=0; j<shoots_per_cane; j++ ){
	
      float xcane = fabs(pos_cane[0].y)+sign*float(j+0.5)*dx;
	
      for( int i=1; i<rad_cane.size(); i++ ){

	if( fabs(pos_cane.at(i-1).y)<=fabs(xcane) && fabs(pos_cane.at(i).y)>=fabs(xcane) ){

	  std::vector<float> rad_pshoot;
	  std::vector<vec3> pos_pshoot;
	  std::vector<RGBcolor> pshoot_color;
	  
	  float frac=(xcane-pos_cane.at(i-1).y)/(pos_cane.at(i).y-pos_cane.at(i-1).y);
	  
	  float Rheight = height*(1+0.25*(1.f-2.f*unif_distribution(generator)));
	
	  //cane base
	  rad_pshoot.push_back(shoot_radius);
	  //pos_pshoot.push_back( make_vec3(xcane, pos_cane.at(i-1).y+frac*(pos_cane.at(i).y-pos_cane.at(i-1).y), pos_cane.at(i-1).z-frac*(pos_cane.at(i).z-pos_cane.at(i-1).z)));
	  pos_pshoot.push_back( make_vec3(pos_cane.at(i-1).x+frac*(pos_cane.at(i).x-pos_cane.at(i-1).x), xcane, pos_cane.at(i-1).z-frac*(pos_cane.at(i).z-pos_cane.at(i-1).z)));
	  pshoot_color.push_back(wood_color);

	  //context->addSphere( 10, origin+pos_pshoot.back(), 0.05 );
	  //context->addSphere( 10, origin+pos_cane.at(i-1), 0.05 );

	  float phirot = (0.5f-unif_distribution(generator))*0.5*M_PI;
	  
	  uint Nz = 10;
	  float dz = (Rheight-trunk_height)/float(Nz);
	  for( uint k=1; k<Nz; k++ ){
	    
	    vec3 n = rotatePoint( make_vec3(0,0,dz), sign*mean_shoot_angle*M_PI/180.f*(1.f-1.2*float(k)/float(Nz-1)), phirot );
	    pos_pshoot.push_back( pos_pshoot.back()+n+make_vec3(-0.5+unif_distribution(generator),-0.5+unif_distribution(generator),-0.5+unif_distribution(generator))*0.1 );
	    
	    rad_pshoot.push_back(shoot_radius);
	    pshoot_color.push_back(wood_color);
	  
	  }

	  std::vector<vec3> tmp;
	  tmp.resize(pos_pshoot.size());
	  for( uint i=0; i<pos_pshoot.size(); i++ ){
	    tmp.at(i) = pos_pshoot.at(i) + origin;
	  }
	  
	  U = context->addTube(8,tmp,rad_pshoot, "plugins/visualizer/textures/wood2.jpg" );
	  UUIDs.push_back(U);
	  context->setPrimitiveData(U,"element_type","shoot");

	  float zleaf = pos_pshoot.back().z;
	  float flip = 0;
	  while( zleaf>pos_pshoot.at(0).z && flip<100 ){

	    for( int i=1; i<rad_pshoot.size(); i++ ){
	      
	      if( pos_pshoot.at(i-1).z<=zleaf && pos_pshoot.at(i).z>=zleaf ){
		
		float lfrac=(zleaf-pos_pshoot.at(i-1).z)/(pos_pshoot.at(i).z-pos_pshoot.at(i-1).z);
		
		vec3 pos_leaf = make_vec3( pos_pshoot.at(i-1).x+lfrac*(pos_pshoot.at(i).x-pos_pshoot.at(i-1).x), pos_pshoot.at(i-1).y+lfrac*(pos_pshoot.at(i).y-pos_pshoot.at(i-1).y), zleaf);
		
		float size = fmaxf(leaf_size*(1.f-exp(-4.f*fabs(pos_pshoot.back().z-zleaf)/(pos_pshoot.back().z-trunk_height))),0.1*leaf_size);
		
		vec3 parent_normal = pos_pshoot.at(i)-pos_pshoot.at(i-1);
		vec3 leaf_offset = rotatePointAboutLine(make_vec3(0,0.25*size,0), make_vec3(0,0,0), parent_normal, flip*M_PI+unif_distribution(generator)*0.25*M_PI );

		float s;
		if( int(flip)%2==0 ){
		  s = 1;
		}else{
		  s = -1;
		}

		float Rphi = orientation + s*0.5*M_PI*(1.f+0.5*(1.f-2.f*unif_distribution(generator)));
		//float Rtheta = 0.5*M_PI*(1.f+0.3*(1.f-2.f*unif_distribution(generator)));

		float Rtheta = sampleLeafAngle(generator);

		vec3 position = origin+pos_leaf+leaf_offset;

		float frac = fabs(cos(2*M_PI*position.x/50.f));

		//context->addAlphaMask( position, make_vec2(size,size), make_SphericalCoord(Rtheta,s*Rphi), cmap.query(frac), "Plugins/visualizer/textures/GrapeLeaf.png" );
		U.resize(0);
		U.push_back( context->addPatch( make_vec3(0,0,0), make_vec2(size,size), make_SphericalCoord(0,0), "plugins/visualizer/textures/GrapeLeaf.png" ) );
	
		context->getPrimitivePointer(U.back())->rotate(Rtheta,"y");
		context->getPrimitivePointer(U.back())->rotate(Rphi,"z");
		context->getPrimitivePointer(U.back())->translate(position);

		UUIDs.push_back(U);
		context->setPrimitiveData(U,"element_type","leaf");
		
		// SphericalCoord rotation = make_SphericalCoord(Rtheta,Rphi);
		// for( int v=0; v<leaf_vertices.size()-1; v++ ){
		//   vec3 v0 = rotatePoint( leaf_vertices.at(0), rotation );
		//   vec3 v1 = rotatePoint( leaf_vertices.at(v), rotation );
		//   vec3 v2 = rotatePoint( leaf_vertices.at(v+1), rotation );
		//   context->addTriangle( position+size*v0, position+size*v1, position+size*v2, leaf_color );
		// }

		zleaf -= 0.65*size*(1+0.25*(0.5f-2.f*unif_distribution(generator)));
		flip++;
		
		break;
		
	      }
	      
	    }
	    
	  }

	  //break;
	  
	}
	
      }
    }
  }

  return UUIDs;

}
