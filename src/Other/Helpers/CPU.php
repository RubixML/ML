<?php

namespace Rubix\ML\Other\Helpers;

class CPU
{
    /**
     * Return the number of cpu cores or 0 if unable to detect.
     *
     * @return int
     */
    public static function cores() : int
    {
        $cores = 0;

        if (is_file('/proc/cpuinfo')) {
            $cpuinfo = file_get_contents('/proc/cpuinfo') ?: '';

            $matches = [];

            preg_match_all('/^processor/m', $cpuinfo, $matches);

            $cores = count($matches[0]);
        }

        return $cores;
    }
}
