import React from 'react';
import PageHeading from './PageHeading';
import Watercolor from './Watercolor';

const Home = (props) => {
  return (<div>
    <PageHeading title="Alphagarden" subtitle="Updated daily from Berkeley, CA" />
    <Watercolor />
  </div>)
}

export default Home;