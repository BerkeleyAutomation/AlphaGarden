#include "Context.h"
#include "WeberPennTree.h"
#include "Visualizer.h"

using namespace helios;

int main(void)
{
    Context context; //declare the context
    // Add data to timeseries
    Date date;
    Time time;
    date = make_Date(2, 1, 2000);                           // 2 Jan. 2000
    time = make_Time(13, 00, 00);                           // 13:00:00
    context.addTimeseriesData("growth", 25.0, date, time);  // index #0
    time = make_Time(13, 15, 00);                           // 13:15:00
    context.addTimeseriesData("growth", 50.0, date, time);  // index #1
    time = make_Time(13, 30, 00);                           // 13:30:00
    context.addTimeseriesData("growth", 75.0, date, time);  // index #2
    time = make_Time(13, 45, 00);                           // 13:45:00
    context.addTimeseriesData("growth", 100.0, date, time); // index #3
    float T;
    T = context.queryTimeseriesData("growth", 1); // Here, T = 50
    time = make_Time(13, 15, 00);
    T = context.queryTimeseriesData("growth", date, time); // Also here, T = 50
    for (uint i = 0; i < context.getTimeseriesLength("growth"); i++)
    {
        T = context.queryTimeseriesData("growth", i);
        time = context.queryTimeseriesTime("growth", i);
        printf("Growth at time %02d:%02d:%02d is %f\n", time.hour, time.minute, time.second, T);
    }
}
