<?php

namespace Rubix\ML\Other\Helpers;

use RuntimeException;
use Exception;

class CPU
{
    protected const CPU_INFO = '/proc/cpuinfo';

    protected const CORE_REGEX = '/^processor/m';

    /**
     * Return the number of cpu cores or 0 if unable to detect.
     *
     * @throws \RuntimeException
     * @return int
     */
    public static function cores() : int
    {
        try {
            $cpuinfo = file_get_contents(self::CPU_INFO);

            if (!$cpuinfo) {
                throw new Exception();
            }

            $matches = [];

            preg_match_all(self::CORE_REGEX, $cpuinfo, $matches);

            $cores = count($matches[0]);
        } catch (Exception $e) {
            throw new RuntimeException('Could not auto detect CPU'
                . ' core count.');
        }

        return $cores;
    }
}
