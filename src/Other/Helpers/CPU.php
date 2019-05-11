<?php

namespace Rubix\ML\Other\Helpers;

class CPU
{
    protected const CPU_INFO = '/proc/cpuinfo';

    protected const CORE_REGEX = '/^processor/m';

    /**
     * Return the number of cpu cores or 0 if unable to detect.
     *
     * @return int
     */
    public static function cores() : int
    {
        $cores = 0;

        if (is_file(self::CPU_INFO)) {
            $cpuinfo = file_get_contents(self::CPU_INFO) ?: '';

            $matches = [];

            preg_match_all(self::CORE_REGEX, $cpuinfo, $matches);

            $cores = count($matches[0]);
        }

        return $cores;
    }
}
