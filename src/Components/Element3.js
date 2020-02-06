import React from 'react';
import POST_ZOOM from './Post_Zoom';
import $ from 'jquery';
import PlantData from '../Media/plant-data';
import Overview from './Overview.js';
import { CSSTransition } from 'react-transition-group';
import ZoomBox1 from '../Media/top-left-border.svg'
import ZoomBox2 from '../Media/top-right-border.svg';
import ZoomBox3 from '../Media/bottom-left-border.svg';
import ZoomBox4 from '../Media/bottom-right-border.svg';
import Grid from '../Media/grid.svg';
import BackVideo from './BackVideo.js'
import RobotCameraOverlay from './RobotCameraOverlay';
import GlowingMarks from './GlowingMarks';
import DateBox from './DateBox';
import ZoomBox from './ZoomBox';
import Overlay from './Overlay';
import TimeoutHelper from './TimeoutHelper';

// Component for dynamic zoom data display

class Element3 extends React.Component{

	constructor(props) {
		super(props);

		this.timer = new TimeoutHelper();

		//Func to trigger the proper transformations to zoom into a square of the garden  Math.floor(Math.random() * 16) + 1
		

		//Func to trigger the proper transformations to zoom into a square of the garden  square=Math.floor(Math.random() * 16) + 1

		const zoomIn = () => {
			var square = Math.floor(Math.random() * 16) + 1;
			while (square === this.state.prevZoomId) {
				square = Math.floor(Math.random() * 16) + 1;
			}
			this.state.prevZoomId = square;

			//calculate which square to zoom into
			if(!this.props.nuc){
				if(this.state.x < 0.25){
					square = 1;
				}
				else if( this.state.x < 0.5){
					square = 2;
				}else if(this.state.x < 0.75){
					square = 3;
				}else{
					square = 4;
				}

				if(this.state.y < 0.25){
					square += 0;
				}else if(this.state.y < 0.5){
					square += 4;
				}else if(this.state.y < 0.75){
					square += 8;
				}else if(this.state.y > 0.75){
					square += 12;
				};
			}
			showBorders(square);
			this.timer.setTimeout(() => {
				removeOverlay();
				triggerZoom(square);
				setZoomPosition(square);
				setOverlay(square);
	
				if(this.props.nuc){
					this.timer.setTimeout(zoomOut, 10000);
				}
			}, 4000)
		}

		// Initialize plant data from JSON
		const GARDEN_WIDTH = 1920;
		const GARDEN_HEIGHT = 1080;
		const GRID_WIDTH = 4;
		const GRID_HEIGHT = 4;

		const gridPlants = {};
		for (let i = 1; i <= GRID_WIDTH * GRID_HEIGHT; i++) {
			gridPlants[i] = [];
		}

		for (let plant of PlantData) {
			const { center_x, center_y } = plant;
			const gridX = Math.floor(center_x / (GARDEN_WIDTH / GRID_WIDTH)) + 1;
			const gridY = Math.floor(center_y / (GARDEN_HEIGHT / GRID_HEIGHT));
			const square = gridX + gridY * GRID_WIDTH;
			gridPlants[square].push(plant);
		}

		console.log(gridPlants);

		const removeOverlay = () => {
			this.setState({
				overlay: null
			});
		}

		const triggerZoom = (box) => {
			this.setState({
				zoom: "Zoom" + box,
				handleClick: zoomOut,
				zoombox1: "zoombox1",
				zoombox2: "zoombox2",
				zoombox3: "zoombox3",
				zoombox4: "zoombox4",
			});
		}

		const showBorders = (box) => {
			this.timer.setTimeout(() => {
				this.setState({
					zoomboximg: true,
				})
			}, 3000);

			var i = box - 1;
			var topLeft = document.getElementById('top-left');
			var topRight = document.getElementById('top-right');
			var bottomLeft = document.getElementById('bottom-left');
			var bottomRight = document.getElementById('bottom-right');
			
			topLeft.style.left = (i % GRID_WIDTH) * (100 / GRID_WIDTH) + '%';
			topLeft.style.top = Math.floor((i / GRID_HEIGHT)) * (100 / GRID_HEIGHT) + '%';

			bottomLeft.style.left = (i % GRID_WIDTH) * (100 / GRID_WIDTH) + '%';
			bottomLeft.style.top = Math.min(Math.floor((i / GRID_HEIGHT)) * (100 / GRID_HEIGHT) + (100 / GRID_HEIGHT), 96) + '%';

			topRight.style.left = Math.min((i % GRID_WIDTH) * (100 / GRID_WIDTH) + (100 / GRID_WIDTH), 96) + '%';
			topRight.style.top = Math.floor((i / GRID_HEIGHT)) * (100 / GRID_HEIGHT) + '%';

			bottomRight.style.left = Math.min((i % GRID_WIDTH) * (100 / GRID_WIDTH) + (100 / GRID_WIDTH), 96) + '%';
			bottomRight.style.top = Math.min(Math.floor((i / GRID_HEIGHT)) * (100 / GRID_HEIGHT) + (100 / GRID_HEIGHT), 96) + '%';
		}

		const setZoomPosition = (box) => {
			var img = document.getElementById('no_zoom');
			if (img == null) {
				img = document.getElementById('ZoomOut');
			}
			if (img == null) {
				img = document.getElementById("Zoom" + box);
			}
			if (img != null) {
				var i = box - 1;
				img.style.transformOrigin = (i % GRID_WIDTH) * (100 / (GRID_WIDTH - 1)) + '%' + Math.floor((i / GRID_HEIGHT)) * (100 / (GRID_HEIGHT - 1))  + '%';
			}
		}

		const setOverlay = (box) => {
			const x = (box - 1) % GRID_WIDTH;
			const y = Math.floor((box - 1) / GRID_WIDTH);

            this.timer.setTimeout(
            	() => {this.setState({
					overlay: <POST_ZOOM box={box} 
								plants={gridPlants[box]} 
								startX={x * (GARDEN_WIDTH / GRID_WIDTH)}
								startY={y * (GARDEN_HEIGHT / GRID_HEIGHT)}
								gridWidth={(GARDEN_WIDTH / GRID_WIDTH)}
								gridHeight={(GARDEN_HEIGHT / GRID_HEIGHT)}
							/>
            	})}
            , 500);
        }
		
		//Func to dynamically zoom back out to the overhead of the garden
		const zoomOut = () => {
			removeOverlay();

			this.setState({
				zoom: "ZoomOut",
				handleClick: zoomIn,
				zoomboximg: false,
				zoombox1: "zoomboxshrink", 
				zoombox2: "zoomboxshrink", 
				zoombox3: "zoomboxshrink",
				zoombox4: "zoomboxshrink"
			})

			this.timer.setTimeout(() => {this.setState({overlay: null})}, 3000);

			if(this.props.nuc){
				this.setState((state) => ({
					counter: state.counter + 1
				}));
				if(this.state.counter >= 2) {
					this.timer.setTimeout(this.props.endFunc, 4000);
				}
				else {
					this.timer.setTimeout(zoomIn, 2000);
				}
			}
		}

    	this.state = {
			overlay: null,
    		zoom: "no_zoom",
    		handleClick: zoomIn,
    		x:0,
    		y:0,
			counter: 0,
			
			prevZoomId: -1,

			overview: false,
			waitToStart: true,

			grid: null,
			robotCameraOverlay: false,
			glowingMarks: false,
			filter: false,

			zoomboximg: false 
    	}
	}
	
   	componentDidMount(){
    }

    //constantly updates the position of
    _onMouseMove(e) {
    	this.setState({ x: (e.clientX / $( window ).width()), y: (e.clientY / $( window ).height())  });
		}	
		
	componentWillUnmount() {
		this.timer.clearAllTimeouts();
	}

	render(){
	  return (
	  		<div onMouseMove={this._onMouseMove.bind(this)}>

				<div id="Zoom_Container">
					<RobotCameraOverlay shouldDisplay={this.state.robotCameraOverlay}/>
					<img src={require("../Media/Garden-Overview.bmp")} alt="GARDEN" height="100%" width="100%" onClick={(e) => {console.log("???"); this.state.handleClick(e)}}  id={this.state.zoom}/>
				</div>

				<CSSTransition
					in={this.state.waitToStart}
					timeout={0}
					unmountOnExit
					onEnter={() => this.setState({waitToStart:false})}
					onExited={() => {this.timer.setTimeout(() => this.setState({overview:true}), 0)}}
					classNames="over"
						>
						<BackVideo id="timelapse-video" vidName={require("../Media/time_lapse.mp4")} endFunc={() => {this.timer.setTimeout(() => this.setState({waitToStart:false}), 200)}} nuc={this.state.nuc}/>
				</CSSTransition>

				<CSSTransition
		        	in={this.state.overview}
		        	timeout={0}
		        	onEnter={() => {this.timer.setTimeout(() => this.setState({overview:false}), 4500)}}
		        	onExited={() => {
						this.setState({
							grid: Grid,
							robotCameraOverlay: true,
							glowingMarks: true,
							filter: true,
						});
						this.state.handleClick();
		   			}}
		        	unmountOnExit
		        	classNames="over"
		        	>
		        	<div>	
						<Overview id="overview"/>
					</div>
      			</CSSTransition>

				<div className="Overlay">
					{this.state.overlay}
				</div>

				<Overlay shouldDisplay={this.state.filter} />

				<div className="ZoomBox" id="top-left">
					<ZoomBox shouldDisplay={this.state.zoomboximg} src={ZoomBox1} id={this.state.zoombox1}/>
				</div>
				<div className="ZoomBox" id="top-right">
					<ZoomBox shouldDisplay={this.state.zoomboximg} src={ZoomBox2} id={this.state.zoombox2}/>
				</div>
				<div className="ZoomBox" id="bottom-left">
					<ZoomBox shouldDisplay={this.state.zoomboximg} src={ZoomBox3} id={this.state.zoombox3}/>
				</div>
				<div className="ZoomBox" id="bottom-right">
					<ZoomBox shouldDisplay={this.state.zoomboximg} src={ZoomBox4} id={this.state.zoombox4}/>
				</div>

				<GlowingMarks shouldDisplay={this.state.glowingMarks} />
				<DateBox shouldDisplay={this.state.robotCameraOverlay} />
		    </div>

	    
	  );
	}
}

export default Element3;