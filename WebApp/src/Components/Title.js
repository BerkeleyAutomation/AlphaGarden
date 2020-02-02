import React from 'react';

 const Title = (props) => {

    if(props.nuc){setTimeout(props.endFunc, 3000)}

	 return(
        <div className="fade-in">
          <div className="IntroDate">
            <p id="jumbotron-date">{props.title}</p>
          </div>
        </div> 
		)    
}

export default Title;