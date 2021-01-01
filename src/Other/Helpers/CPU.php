<?php

namespace Rubix\ML\Other\Helpers;

use Rubix\ML\Exceptions\RuntimeException;

use function count;

/**
 * CPU
 *
 * @internal
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CPU
{
    /**
     * The command to return the number of processor cores on Windows OS.
     *
     * @var string
     */
    protected const WIN_CORES = 'wmic cpu get NumberOfCores';

    /**
     * The command to return the number of processor cores on Linux.
     *
     * @var string
     */
    protected const CPU_INFO = '/proc/cpuinfo';

    /**
     * The regular expression used to extract the core count.
     *
     * @var string
     */
    protected const CORE_REGEX = '/^processor/m';

    /**
     * Return the number of cpu cores or 0 if unable to detect.
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return int
     */
    public static function cores() : int
    {
        switch (true) {
            case stripos(strtolower(PHP_OS), 'win') === 0:
                $results = explode("\n", shell_exec(self::WIN_CORES) ?: '');

                return (int) preg_replace('/[^0-9]/', '', $results[1]);

            case is_readable(self::CPU_INFO):
                $cpuinfo = file_get_contents(self::CPU_INFO) ?: '';

                $matches = [];

                preg_match_all(self::CORE_REGEX, $cpuinfo, $matches);

                return count($matches[0]);

            default:
                throw new RuntimeException('Could not detect number'
                    . ' of processor cores.');
        }
    }
}
