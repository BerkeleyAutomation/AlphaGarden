import React from 'react';
import Post_Zoom from './Post_Zoom';
import $ from 'jquery';
import PlantData from '../Media/plant-data';
import ZoomBox1 from '../Media/top-left-border.svg';
import ZoomBox2 from '../Media/top-right-border.svg';
import ZoomBox3 from '../Media/bottom-left-border.svg';
import ZoomBox4 from '../Media/bottom-right-border.svg';
import Grid from '../Media/grid.svg';
import RobotCameraOverlay from './RobotCameraOverlay';
import GlowingMarks from './GlowingMarks';
import DateBox from './DateBox';
import ZoomBox from './ZoomBox';
import Overlay from './Overlay';

// Initialize plant data from JSON
const GARDEN_WIDTH = 1920;
const GARDEN_HEIGHT = 1080;
const GRID_WIDTH = 4;
const GRID_HEIGHT = 4;

// Component for dynamic zoom data display

class Element3 extends React.Component {
  constructor(props) {
    super(props);

    const gridPlants = {};
    for (let i = 1; i <= GRID_WIDTH * GRID_HEIGHT; i++) {
      gridPlants[i] = [];
    }

    // allocate plants to 1 of 16 grids.
    // Resize doesn't matter because center_x,center_y is has x,y positions
    // relative to the original dimensions of the picture
    for (let plant of PlantData) {
      const { center_x, center_y } = plant;
      const gridX = Math.floor(center_x / (GARDEN_WIDTH / GRID_WIDTH)) + 1;
      const gridY = Math.floor(center_y / (GARDEN_HEIGHT / GRID_HEIGHT));
      const square = gridX + gridY * GRID_WIDTH;
      gridPlants[square].push(plant);
    }

    console.log(gridPlants);

    this.state = {
      overlay: null,
      zoom: 'no_zoom',
      handleClick: () => this.zoomIn(),
      x: 0,
      y: 0,
      counter: 0,

      prevZoomId: -1,

      overview: false,
      waitToStart: true,

      grid: Grid,
      robotCameraOverlay: true,
      glowingMarks: true,
      filter: true,

      zoomboximg: false,
      gridPlants: gridPlants,
      width: window.innerWidth,
      height: window.innerHeight
    };

    this.zoomIn = this.zoomIn.bind(this);
    this.zoomOut = this.zoomOut.bind(this);
    this.setOverlay = this.setOverlay.bind(this);
  }

  showBorders(box) {
    setTimeout(() => {
      this.setState({
        zoomboximg: true
      });
    }, 0);

    var i = box - 1;
    var topLeft = document.getElementById('top-left');
    var topRight = document.getElementById('top-right');
    var bottomLeft = document.getElementById('bottom-left');
    var bottomRight = document.getElementById('bottom-right');

    topLeft.style.left = (i % GRID_WIDTH) * (100 / GRID_WIDTH) + '%';
    topLeft.style.top = Math.floor(i / GRID_HEIGHT) * (100 / GRID_HEIGHT) + '%';

    bottomLeft.style.left = (i % GRID_WIDTH) * (100 / GRID_WIDTH) + '%';
    bottomLeft.style.top =
      Math.min(
        Math.floor(i / GRID_HEIGHT) * (100 / GRID_HEIGHT) + 100 / GRID_HEIGHT,
        96
      ) + '%';

    topRight.style.left =
      Math.min((i % GRID_WIDTH) * (100 / GRID_WIDTH) + 100 / GRID_WIDTH, 96) +
      '%';
    topRight.style.top =
      Math.floor(i / GRID_HEIGHT) * (100 / GRID_HEIGHT) + '%';

    bottomRight.style.left =
      Math.min((i % GRID_WIDTH) * (100 / GRID_WIDTH) + 100 / GRID_WIDTH, 96) +
      '%';
    bottomRight.style.top =
      Math.min(
        Math.floor(i / GRID_HEIGHT) * (100 / GRID_HEIGHT) + 100 / GRID_HEIGHT,
        96
      ) + '%';
  }

  //Func to trigger the proper transformations to zoom into a square of the garden  square=Math.floor(Math.random() * 16) + 1
  zoomIn() {
    this.setState({
      handleClick: () => {}
    });
    var square;
    if (this.state.x < 0.25) {
      square = 1;
    } else if (this.state.x < 0.5) {
      square = 2;
    } else if (this.state.x < 0.75) {
      square = 3;
    } else {
      square = 4;
    }

    if (this.state.y < 0.25) {
      square += 0;
    } else if (this.state.y < 0.5) {
      square += 4;
    } else if (this.state.y < 0.75) {
      square += 8;
    } else if (this.state.y > 0.75) {
      square += 12;
    }
    console.log(this.state.x, this.state.y, square);
    this.showBorders(square);
    setTimeout(() => {
      this.removeOverlay();
      this.triggerZoom(square);
      this.setZoomPosition(square);
      this.setOverlay(square);

      if (this.props.nuc) {
        setTimeout(this.zoomOut, 10000);
      }
    }, 1000);
  }

  removeOverlay() {
    this.setState({
      overlay: null
    });
  }

  triggerZoom(box) {
    this.setState({
      zoom: 'Zoom' + box,
      handleClick: () => {},
      zoombox1: 'zoombox1',
      zoombox2: 'zoombox2',
      zoombox3: 'zoombox3',
      zoombox4: 'zoombox4'
    });
  }

  setZoomPosition(box) {
    var img = document.getElementById('no_zoom');
    if (img == null) {
      img = document.getElementById('ZoomOut');
    }
    if (img == null) {
      img = document.getElementById('Zoom' + box);
    }
    if (img != null) {
      var i = box - 1;
      img.style.transformOrigin =
        (i % GRID_WIDTH) * (100 / (GRID_WIDTH - 1)) +
        '%' +
        Math.floor(i / GRID_HEIGHT) * (100 / (GRID_HEIGHT - 1)) +
        '%';
      console.log(img.style.transformOrigin);
    }
  }

  setOverlay(box) {
    const { width, height, gridPlants } = this.state;
    // const proportion = width / GARDEN_WIDTH;

    const x = (box - 1) % GRID_WIDTH;
    const y = Math.floor((box - 1) / GRID_WIDTH);

    const startX = x * (width / GRID_WIDTH);
    const startY = y * (height / GRID_HEIGHT);

    setTimeout(() => {
      this.setState({
        overlay: (
          <Post_Zoom
            box={box}
            plants={gridPlants[box]}
            startX={startX}
            startY={startY}
            gridWidth={width / GRID_WIDTH}
            gridHeight={height / GRID_HEIGHT}
          />
        )
      });
    }, 500);
  }

  //Func to dynamically zoom back out to the overhead of the garden
  zoomOut() {
    this.removeOverlay();

    this.setState({
      zoom: 'ZoomOut',
      zoomboximg: false,
      zoombox1: 'zoomboxshrink',
      zoombox2: 'zoomboxshrink',
      zoombox3: 'zoomboxshrink',
      zoombox4: 'zoomboxshrink'
    });

    setTimeout(() => {
      this.setState({ overlay: null, handleClick: this.zoomIn });
    }, 3000);

    if (this.props.nuc) {
      this.setState(state => ({
        counter: state.counter + 1
      }));
      if (this.state.counter >= 2) {
        setTimeout(this.props.endFunc, 4000);
      } else {
        setTimeout(this.zoomIn, 2000);
      }
    }
  }

  updateDimensions = () => {
    this.setState({
      width: window.innerWidth,
      height: window.innerHeight
    });
  };

  componentDidMount() {
    window.addEventListener('resize', this.updateDimensions);
  }

  componentWillUnmount() {
    window.removeEventListener('resize', this.updateDimensions);
  }

  //constantly updates the position of
  _onMouseMove(e) {
    const { width } = this.state;
    const proportion = width / GARDEN_WIDTH;
    this.setState({
      x: e.clientX / (proportion * GARDEN_WIDTH),
      y: e.clientY / (proportion * GARDEN_HEIGHT)
      // x: (e.clientX / $(window).width()),
      // y: (e.clientY / $(window).height())
    });
    // console.log(e.clientX / $( window ).width() + ", " + e.clientY / $( window ).height());
  }

  render() {
    const { width, height } = this.state;
    // if in portrait mode
    if (height > width) {
      return (
        <div>
          <div id="Static_Mobile_Text">
            Please turn your phone to landscape mode for the optimal interactive
            experience
          </div>
          <div id="Static_Zoom_Container">
            <img
              src={require('../Media/Garden-Overview.bmp')}
              alt="GARDEN"
              height="100%"
              width="100%"
              id={this.state.zoom}
            />
          </div>
        </div>
      );
    }
    return (
      <div onMouseMove={this._onMouseMove.bind(this)}>
        <div id="Zoom_Container">
          <img
            src={require('../Media/Garden-Overview.bmp')}
            alt="GARDEN"
            height="100%"
            width="100%"
            id={this.state.zoom}
          />
          <RobotCameraOverlay shouldDisplay={this.state.robotCameraOverlay} />
        </div>

        <div className="Overlay">{this.state.overlay}</div>

        <Overlay shouldDisplay={this.state.filter} />

        <div className="ZoomBox" id="top-left">
          <ZoomBox
            shouldDisplay={this.state.zoomboximg}
            src={ZoomBox1}
            id={this.state.zoombox1}
          />
        </div>
        <div className="ZoomBox" id="top-right">
          <ZoomBox
            shouldDisplay={this.state.zoomboximg}
            src={ZoomBox2}
            id={this.state.zoombox2}
          />
        </div>
        <div className="ZoomBox" id="bottom-left">
          <ZoomBox
            shouldDisplay={this.state.zoomboximg}
            src={ZoomBox3}
            id={this.state.zoombox3}
          />
        </div>
        <div className="ZoomBox" id="bottom-right">
          <ZoomBox
            shouldDisplay={this.state.zoomboximg}
            src={ZoomBox4}
            id={this.state.zoombox4}
          />
        </div>

        <GlowingMarks shouldDisplay={this.state.glowingMarks} />
        <DateBox shouldDisplay={this.state.robotCameraOverlay} />
        {/* <RobotCameraOverlay shouldDisplay={this.state.robotCameraOverlay}/> */}
        <div
          id="click"
          onClick={e => {
            console.log('clicked');
            this.state.handleClick(e);
          }}
        ></div>
      </div>
    );
  }
}

export default Element3;
