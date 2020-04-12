import React from 'react';
import '../Website.css';
import spinner from '../Media/spinner.png';

const LoadingComponent = props => {
  return (
    <div className="loading-component">
      <img
        src={spinner}
        className={props.isPortrait ? "loading-component__rotate--portrait" : "loading-component__rotate"}
        alt="video loading"
      />
      {/* <p>Loading Video...</p> */}
    </div>
  );
};

export default LoadingComponent;
