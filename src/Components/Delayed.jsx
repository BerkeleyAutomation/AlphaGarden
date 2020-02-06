import React from 'react';
import PropTypes from 'prop-types';
import TimeoutHelper from './TimeoutHelper';

class Delayed extends React.Component {

    constructor(props) {
        super(props);
        this.state = {hidden : true};
        this.timer = new TimeoutHelper();
    }

    componentDidMount() {
        this.timer.setTimeout(() => {
            this.setState({hidden: false});
        }, this.props.waitBeforeShow);
    }

    componentWillUnmount() {
        this.timer.clearAllTimeouts();
    }

    render() {
        return this.state.hidden ? '' : this.props.children;
    }
}

Delayed.propTypes = {
  waitBeforeShow: PropTypes.number.isRequired
};

export default Delayed;