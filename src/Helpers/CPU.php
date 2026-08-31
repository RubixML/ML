<?php

namespace Rubix\ML\Helpers;

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
     * @var literal-string
     */
    protected const WIN_CORES = 'wmic cpu get NumberOfCores';

    /**
     * The command to return the number of processor cores on Linux.
     *
     * @var literal-string
     */
    protected const CPU_INFO = '/proc/cpuinfo';

    /**
     * The regular expression used to split the cpuinfo output into blocks.
     *
     * @var literal-string
     */
    protected const PROCESSOR_REGEX = '/\n(?=processor\s*:)/';

    /**
     * The cached machine epsilon.
     *
     * @var float|null
     */
    protected static ?float $epsilon = null;

    /**
     * Return the number of physical cpu cores or 0 if unable to detect.
     *
     * @throws RuntimeException
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

                return self::extractPhysicalCoreCount($cpuinfo);

            default:
                throw new RuntimeException('Could not detect number'
                    . ' of processor cores.');
        }
    }

    /**
     * Return the estimated machine epsilon.
     *
     * @return float
     */
    public static function epsilon() : float
    {
        if (self::$epsilon === null) {
            $epsilon = $previous = 1.0;

            while (1.0 + $epsilon !== 1.0) {
                $previous = $epsilon;

                $epsilon *= 0.5;
            }

            self::$epsilon = $previous;
        }

        return self::$epsilon;
    }

    /**
     * Count the number of unique physical cores in the cpuinfo contents,
     * falling back to the logical core count if core ids are unavailable.
     *
     * @param string $cpuinfo
     * @return int
     */
    protected static function extractPhysicalCoreCount(string $cpuinfo) : int
    {
        $cores = [];
        $logical = 0;

        foreach (preg_split(self::PROCESSOR_REGEX, $cpuinfo) as $block) {
            if (preg_match('/^processor\s*:/m', $block) !== 1) {
                continue;
            }

            $physical = self::parseId($block, 'physical id');
            $core = self::parseId($block, 'core id');

            if ($core === null) {
                ++$logical;

                continue;
            }

            $cores["{$physical}-{$core}"] = true;
        }

        return $cores !== [] ? count($cores) : $logical;
    }

    /**
     * Parse a single identifier attribute from a cpuinfo block or null if absent.
     *
     * @param string $block
     * @param string $attribute
     * @return int|null
     */
    protected static function parseId(string $block, string $attribute) : ?int
    {
        $matches = [];

        $pattern = '/^\s*' . $attribute . '\s*:\s*(\d+)/m';

        if (preg_match($pattern, $block, $matches) !== 1) {
            return null;
        }

        return (int) $matches[1];
    }
}
