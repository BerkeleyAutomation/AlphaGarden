import React from 'react';
// Component that sets background video. Props: {videoName, endFunc}
class BackVideo extends React.Component{

	render(){
	  return (
			<video autoPlay muted loop id="backgroundVideo" onEnded={this.props.endFunc} height="100%" width="100%">
				<source src={this.props.vidName}  />
			</video>
	  );
	}
}

export default BackVideo;