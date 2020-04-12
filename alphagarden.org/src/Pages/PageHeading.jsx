import React from 'react';

const PageHeading = ({title, subtitle, isPortrait}) => {
  return (<div className="page-heading">
    <h1 className={isPortrait ? "page-heading__title--portrait" : "page-heading__title"}>{title}</h1>
    <h2 className={isPortrait ? "page-heading__subtitle--portrait" : "page-heading__subtitle"}>{subtitle}</h2>
  </div>)
}

export default PageHeading;