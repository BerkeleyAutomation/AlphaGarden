import React from 'react';
// Component that sets background video. Props: {videoName, endFunc}
class BackVideo extends React.Component{

	render(){
	  return (
	    
		 <video autoPlay muted id="backgroundVideo" onEnded={this.props.endFunc}>
	        <source src={this.props.vidName}  />
	     < /video>

	    
	  );
	}
}

export default BackVideo;