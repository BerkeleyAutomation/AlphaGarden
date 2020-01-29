import React from 'react';
import GlowingMarks from './GlowingMarks'
	
	let diversity = "36%";
	let growth = "50%";
	let plants = ["Lavendar", "Basil", "Turnips", "Rosemary", "Cacti"];
	const plantMap = (x) => {return x + ", "}
 //component for the text that goes over the background video
 const Overview = (props) => {

 	if(props.nuc){setTimeout(props.endFunc, 6000)}

	let today = new Date();
	 return(
 	 	<div className = "LOADING">
 	 		{ <GlowingMarks /> }
	 		<img src={require("./Garden-Overview2.bmp")} alt="Zaaa GARDEN" height="100%" width="100%" />

	    	<div className="Overlay1">
	    		<p className="top"> ALPHA GARDEN</p>
		 		<button id="button" onClick={props.endFunc}> Day  {today.getDate() + today.getMonth()} </button>
		 	</div>
		 	<div className = "Data">
		 			<p> Growth: +{growth} <br/> Diversity: +{diversity} </p>
		 	</div>
		</div>
		)    
}

export default Overview;