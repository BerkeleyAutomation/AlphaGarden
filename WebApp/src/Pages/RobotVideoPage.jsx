import React from 'react';
import PageHeading from './PageHeading';
import BackVideo from '../Components/BackVideo';

const RobotVideoPage = (props) => {
  const { windowWidth, windowHeight } = props;
  return (<div>
    <PageHeading
      title="Alphagarden"
      subtitle="Robot"
      isPortrait={windowHeight > windowWidth}
    />
    <BackVideo
      vidName={require("../Media/robot_full.mp4")}
      endFunc={() => { }}
      isPortrait={windowHeight > windowWidth}
    />
  </div>)
}

export default RobotVideoPage;