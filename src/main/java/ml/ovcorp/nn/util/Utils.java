package ml.ovcorp.nn.util;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

public class Utils {

    public static ResultTime diffLocalDateTime(LocalDateTime from, LocalDateTime to) {

        LocalDateTime tempDateTime = LocalDateTime.from(from);

        long years = tempDateTime.until(to, ChronoUnit.YEARS);
        tempDateTime = tempDateTime.plusYears(years);

        long months = tempDateTime.until(to, ChronoUnit.MONTHS);
        tempDateTime = tempDateTime.plusMonths(months);

        long days = tempDateTime.until(to, ChronoUnit.DAYS);
        tempDateTime = tempDateTime.plusDays(days);

        long hours = tempDateTime.until(to, ChronoUnit.HOURS);
        tempDateTime = tempDateTime.plusHours(hours);

        long minutes = tempDateTime.until(to, ChronoUnit.MINUTES);
        tempDateTime = tempDateTime.plusMinutes(minutes);

        long seconds = tempDateTime.until(to, ChronoUnit.SECONDS);
        tempDateTime = tempDateTime.plusSeconds(seconds);

        long milliseconds = tempDateTime.until(to, ChronoUnit.MILLIS);

        return new ResultTime(years, months, days, hours, minutes, seconds, milliseconds);
    }

    public static class ResultTime {

        private final long years;
        private final long months;
        private final long days;
        private final long hours;
        private final long minutes;
        private final long seconds;
        private final long milliseconds;

        public ResultTime(long years, long months, long days, long hours, long minutes, long seconds, long milliseconds) {
            this.years = years;
            this.months = months;
            this.days = days;
            this.hours = hours;
            this.minutes = minutes;
            this.seconds = seconds;
            this.milliseconds = milliseconds;
        }

        public long getYears() {
            return years;
        }

        public long getMonths() {
            return months;
        }

        public long getDays() {
            return days;
        }

        public long getHours() {
            return hours;
        }

        public long getMinutes() {
            return minutes;
        }

        public long getSeconds() {
            return seconds;
        }

        public long getMilliseconds() {
            return milliseconds;
        }

        @Override
        public String toString() {
            StringBuilder timeString = new StringBuilder();

            if (years > 0) {
                timeString.append(years).append(" y ");
            }

            if (months > 0) {
                timeString.append(months).append(" mn ");
            }

            if (days > 0) {
                timeString.append(days).append(" d ");
            }

            if (hours > 0) {
                timeString.append(hours).append(" h ");
            }

            if (minutes > 0) {
                timeString.append(minutes).append(" m ");
            }

            if (seconds > 0) {
                timeString.append(seconds).append(" s ");
            }

            if (milliseconds > 0) {
                timeString.append(milliseconds).append(" ms ");
            }

            return timeString.toString();
        }
    }
}
