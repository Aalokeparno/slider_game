classdef PWLF

    % addpath '/Users/wht02001/Dropbox/Matlab/Mfiles/PiecewiseLinear'

    properties
        bseq
        fmat
        ncurves
    end

    methods
        function obj = PWLF(fmat, bseq)

            % CONSTRUCTOR

            if nargin == 2
                obj.fmat = fmat;
                obj.bseq = bseq;
                obj.ncurves = 0;
            end

        end

        function yy = eval(obj, xx)

            % PWEVAL return the value of the piecewise linear function at
            % each input in xx;  xx should be one-dimensional

            npoints = length(xx);
            yy = zeros(npoints, obj.ncurves);
            for pcount = 1:npoints
                bpoint = find(obj.bseq > xx(pcount));
                currind = bpoint(1);
                for ccount = 1:obj.ncurves
                    yy(pcount, ccount) = obj.fmat(currind, 1, ccount)*xx(pcount) + obj.fmat(currind, 2, ccount);
                end
            end
        end

        function obj = buildf(obj, datamat, xvals)

            % BUILDPWL build a continuous piecewise linear function by
            % linearly interpreting between the points in datamat (each row
            % is a point; each column is a series of y-vals).  Optionally 
            % specify input vals in xvals

            nsections = size(datamat, 1) - 1;
            obj.ncurves = size(datamat, 2);

            if nargin == 2
                xvals = (1:(nsections+1))';
            end

            obj.bseq = xvals(2:end);
            obj.bseq(end) = Inf;  % extrapolate all the way rightward

            obj.fmat = zeros(nsections, 2, obj.ncurves);

            xdiffs = diff(xvals);

            ydiffs = diff(datamat); % diff does diff of the columns

            for ccount = 1:obj.ncurves
                obj.fmat(:, 1, ccount) = ydiffs(:, ccount) ./ xdiffs;
                obj.fmat(:, 2, ccount) = datamat(2:end, ccount) - obj.fmat(:, 1, ccount).*xvals(2:end);
            end
        end
    end
end

