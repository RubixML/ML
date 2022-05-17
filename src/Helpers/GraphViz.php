<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Exceptions\RuntimeException;

use function tempnam;
use function system;
use function escapeshellarg;
use function sys_get_temp_dir;
use function file_put_contents;
use function unlink;

/**
 * GraphViz
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class GraphViz
{
    /**
     * Produces an image from a "dot" formatted string.
     *
     * @param string $dot
     * @param string $path
     * @param string $format
     * @throws \Rubix\ML\Exceptions\RuntimeException
     */
    public static function dotToImage(string $dot, string $path, string $format = 'png') : void
    {
        $tempPath = tempnam(sys_get_temp_dir(), 'graphviz.dot.');

        if ($tempPath === false) {
            throw new RuntimeException('Unable to get temporary filename.');
        }

        $success = file_put_contents($tempPath, $dot, LOCK_EX);

        if ($success === false) {
            throw new RuntimeException('Unable to write to temporary file.');
        }

        $command = 'dot -T ' . escapeshellarg($format)
            . ' ' . escapeshellarg($tempPath)
            . ' -o ' . escapeshellarg($path);

        $ret = 0;

        system($command, $ret);

        unlink($tempPath);

        if ($ret !== 0) {
            throw new RuntimeException("Failed to create image file '$path' (code $ret).");
        }
    }
}
