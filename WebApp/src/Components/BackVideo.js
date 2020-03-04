import React from 'react';
import LoadingComponent from './LoadingComponent';
// Component that sets background video. Props: {videoName, endFunc}
class BackVideo extends React.Component {
  constructor(props) {
    super(props);
    this.onProgress = this.onProgress.bind(this);
    this.state = {
      isLoading: true
    };
  }

  onProgress = evt => {
    // console.log(lengthComputable, total, loaded);
    // console.log(params);
  };

	render() {
		const { isLoading } = this.state;
    return (
		<div>
			{isLoading ? <LoadingComponent isPortrait={this.props.isPortrait} /> : null}
        <video
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          className="back-video"
          onEnded={this.props.endFunc}
          height="100%"
          width="100%"
		  onProgress={evt => {
			  console.log(evt.nativeEvent);
		  }}
		  onLoadedData={() => {
				this.setState({ isLoading: false });
			}}
        >
          <source src={this.props.vidName} type="video/mp4" />
		</video>
      </div>
    );
  }
}

export default BackVideo;
