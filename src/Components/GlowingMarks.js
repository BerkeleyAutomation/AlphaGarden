import React from 'react';
import Grid from '../Media/grid.svg'
	
 const GlowingMarks = (props) => {
	if (props.shouldDisplay == true) {
		return(
			<img src={Grid} className="GridOverlay" />
		)
	} else {
		return null;
	}
}

export default GlowingMarks;