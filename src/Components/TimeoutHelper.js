class TimeoutHelper {
  constructor() {
    this.timers = {};
  }

  timerFinished(id) {
    if (this.timers[id]) {
      delete this.timers[id];
    }
  }

  setTimeout(callback, delay) {
    const id = window.setTimeout(() => {
      callback();
      this.timerFinished(id);
    }, delay)
    this.timers[id] = true;
  }

  clearAllTimeouts() {
    for (const id of Object.keys(this.timers)) {
      clearTimeout(id);
    }
  }
}

export default TimeoutHelper;