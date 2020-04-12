import React from 'react';
import Grid from '../Media/zoom_grid.svg'
	
 const GlowingMarks = (props) => {
	if (props.shouldDisplay === true) {
		return(
			<img src={Grid} className="GridOverlay" alt="grid overlay"/>
		)
	} else {
		return null;
	}
}

export default GlowingMarks;