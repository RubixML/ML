<?php

namespace Rubix\ML\Helpers;

use Rubix\ML\Encoding;
use Rubix\ML\Exceptions\RuntimeException;

use function is_resource;
use function proc_open;
use function proc_close;
use function fwrite;
use function fclose;
use function stream_get_contents;

/**
 * Graphviz
 *
 * An interface to the popular Graphviz program for generating graph images.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 */
class Graphviz
{
    /**
     * Produces an image from a "dot" formatted string.
     *
     * https://graphviz.org/doc/info/lang.html
     *
     * See https://graphviz.org/docs/outputs/ for supported formats
     *
     * @param Encoding $dot
     * @param string $format
     * @throws RuntimeException
     * @return Encoding
     */
    public static function dotToImage(Encoding $dot, string $format = 'png') : Encoding
    {
        $command = "dot -T{$format}";

        $descriptors = [
            ['pipe', 'r'],
            ['pipe', 'w'],
            ['pipe', 'w'],
        ];

        $pipes = [];

        $process = proc_open($command, $descriptors, $pipes);

        if (!is_resource($process)) {
            throw new RuntimeException('Graphviz process could not be opened.');
        }

        fwrite($pipes[0], $dot);
        fclose($pipes[0]);

        $data = stream_get_contents($pipes[1]);
        fclose($pipes[1]);

        $error = stream_get_contents($pipes[2]);
        fclose($pipes[2]);

        if ($error or !$data) {
            throw new RuntimeException("Graphviz encountered an error. $error");
        }

        $ret = proc_close($process);

        if ($ret !== 0) {
            throw new RuntimeException("Graphviz failed to execute (code $ret).");
        }

        return new Encoding($data);
    }
}
