import React from 'react';

const PageHeading = ({title, subtitle}) => {
  return (<div className="page-heading">
    <h1 className="page-heading__title">{title}</h1>
    <h2 className="page-heading__subtitle">{subtitle}</h2>
  </div>)
}

export default PageHeading;