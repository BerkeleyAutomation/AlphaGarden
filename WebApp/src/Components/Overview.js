import React from 'react';
import GlowingMarks from './GlowingMarks';
import Typing from './Typing';
	
	let diversity = "36%";
	let growth = "50%";
	let plants = ["Lavendar", "Basil", "Turnips", "Rosemary", "Cacti"];
	const plantMap = (x) => {return x + ", "}
 //component for the text that goes over the background video
 const Overview = (props) => {

 	if(props.nuc){setTimeout(props.endFunc, props.duration)}

	let today = new Date();
	 return(
 	 	<div className = "LOADING">
			<div id="boxed">
				<h1> DAY: {today.getDate() + today.getMonth()}</h1>
				{/* <h1> COVERAGE: {growth} </h1>  */}
				{/* <h1>DIVERSITY: {diversity} </h1> */}
		 	</div> 
		</div>
		)    
}

export default Overview;