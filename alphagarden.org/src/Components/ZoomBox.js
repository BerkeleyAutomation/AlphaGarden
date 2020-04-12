import React from 'react';
	
 const ZoomBox = (props) => {
	if (props.shouldDisplay === true) {
		return(
			<img src={props.src} id={props.id} alt="zoom frame corner"/>
		)
	} else {
		return null;
	}
}

export default ZoomBox;