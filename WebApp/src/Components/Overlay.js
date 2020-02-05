import React from 'react';
	
 const ZoomBox = (props) => {
	if (props.shouldDisplay === true) {
		return(
            <div className="overlay"></div>
		)
	} else {
		return null;
	}
}

export default ZoomBox;