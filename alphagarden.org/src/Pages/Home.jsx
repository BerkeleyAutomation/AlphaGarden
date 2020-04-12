import React from 'react';
import PageHeading from './PageHeading';
import Watercolor from './Watercolor';

const Home = (props) => {
  const { windowWidth, windowHeight } = props;
  return (<div>
    <PageHeading
      title="Alphagarden"
      subtitle="Updated daily from Berkeley, CA"
      isPortrait={windowHeight > windowWidth}
    />
    <Watercolor
      isPortrait={windowHeight > windowWidth}
    />
  </div>)
}

export default Home;