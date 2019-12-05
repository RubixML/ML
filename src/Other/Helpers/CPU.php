<?php

namespace Rubix\ML\Other\Helpers;

use RuntimeException;
use Exception;

use function count;

class CPU
{
    protected const WIN_CORES = 'wmic cpu get NumberOfLogicalProcessors';
    
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
            switch (true) {
                case stripos(PHP_OS, 'WIN') === 0:
                    $results = explode("\n", shell_exec(self::WIN_CORES) ?? '');

                    if (empty($results)) {
                        throw new Exception();
                    }

                    return (int) preg_replace('/[^0-9]/', '', $results[1]);

                case is_readable(self::CPU_INFO):
                    $cpuinfo = file_get_contents(self::CPU_INFO) ?: '';

                    $matches = [];

                    preg_match_all(self::CORE_REGEX, $cpuinfo, $matches);

                    $cores = count($matches[0]);

                    if ($cores < 1) {
                        throw new Exception();
                    }

                    return $cores;

                default:
                    throw new Exception();
            }
        } catch (Exception $e) {
            throw new RuntimeException('Could not detect processor core count.');
        }
    }
}
