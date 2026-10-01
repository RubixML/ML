<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Set;

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
     * The command to return the number of processor cores on macOS and BSD.
     *
     * @var literal-string
     */
    protected const SYSCTL_CORES = 'sysctl -n hw.physicalcpu';

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
     * Return the number of physical cpu cores or null if unable to detect.
     *
     * @return int|null
     */
    public static function cores() : ?int
    {
        if (str_starts_with(strtolower(PHP_OS), 'win')) {
            return self::windowsCores();
        }

        if (is_readable(self::CPU_INFO)) {
            return self::extractPhysicalCoreCount(file_get_contents(self::CPU_INFO) ?: '') ?: null;
        }

        return self::sysctlCores();
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
     * Return the number of physical cpu cores reported by the Windows
     * management instrumentation command or null if unable to detect.
     *
     * @return int|null
     */
    protected static function windowsCores() : ?int
    {
        $results = explode("\n", shell_exec(self::WIN_CORES) ?: '');

        return self::parseCount($results[1] ?? '');
    }

    /**
     * Return the number of physical cpu cores reported by the system control
     * command or null if unable to detect.
     *
     * @return int|null
     */
    protected static function sysctlCores() : ?int
    {
        return self::parseCount(shell_exec(self::SYSCTL_CORES) ?: '');
    }

    /**
     * Parse the core count from the output of a shell command or null if the
     * output does not contain one.
     *
     * @param string $output
     * @return int|null
     */
    protected static function parseCount(string $output) : ?int
    {
        $count = (int) preg_replace('/[^0-9]/', '', $output);

        return $count > 0 ? $count : null;
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
        $cores = new Set();
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

            $cores->add("{$physical}-{$core}");
        }

        return $cores->count() ?: $logical;
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
