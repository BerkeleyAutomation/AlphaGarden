import React from 'react';
import PageHeading from './PageHeading';
import BackVideo from '../Components/BackVideo';

const RobotVideoPage = () => {
  return (<div>
    <PageHeading title="Alphagarden" subtitle="Robot" />
    <BackVideo vidName={require("../Media/robot_full.mp4")} endFunc={() => {}}/>
  </div>)
}

export default RobotVideoPage;