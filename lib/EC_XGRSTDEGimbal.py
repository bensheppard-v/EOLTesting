import numpy as np
from zaber_motion import Units, Library
from zaber_motion.ascii import Connection

# connect = Connection.open_serial_port('COM6')
# devices = connect.detect_devices()
# dev0 = devices[0]
# dev1 = devices[1]

class EC_XGRSTDEGimbal:
    def __init__(self,port_name,should_home=False):
        self._connection = Connection.open_serial_port(port_name)
        self._devices = self._connection.detect_devices()
        self._verticalmotor = self._devices[0]
        self._horizontalmotor = self._devices[1]
        self.verticalAxis = self._verticalmotor.get_axis(1)
        self.horizontalAxis = self._horizontalmotor.get_axis(1)
        self.axis_unit = Units.ANGLE_DEGREES
        self.horizontalLimits = None
        self.verticalLimits=None


    def home_both_axes(self):
        """
        homes both axes simultaneously, only returns when done.
        """
        self.home_vertical_axis(False)
        self.home_horizontal_axis(False)
        h_status = True
        v_status = True
        while h_status|v_status:
            h_status=self.is_horizontal_axis_busy()
            v_status=self.is_vertical_axis_busy()

    def voyant_home_both_axes(self):
        self.voyant_home_horizontal_axis(False)
        self.voyant_home_vertical_axis(False)
        h_status = True
        v_status = True
        while h_status|v_status:
            h_status=self.is_horizontal_axis_busy()
            v_status=self.is_vertical_axis_busy()
        self.home_horizontal_axis()
        self.home_vertical_axis()

    def home_vertical_axis(self,wait_til_idle=True):
        """
        home always moves in descending direction
        """
        self.verticalAxis.home(wait_until_idle=wait_til_idle)

    def voyant_home_vertical_axis(self,wait_til_idle=True):
        v_pos = self.get_vertical_axis_position()
        if v_pos%360>180:
            move_past = 360-(v_pos%360)+.05
            self.move_vertical_axis_relative(move_past,wait_til_idle=wait_til_idle)
            # v_busy = True
            # while v_busy:
            #     v_busy = self.is_vertical_axis_busy()
        if wait_til_idle | (v_pos%360<=180):
            self.home_vertical_axis(wait_til_idle=wait_til_idle)

    def home_horizontal_axis(self,wait_til_idle=True):
        """
        home always moves in descending direction
        """
        self.horizontalAxis.home(wait_until_idle=wait_til_idle)

    def voyant_home_horizontal_axis(self,wait_til_idle=True):
        h_pos = self.get_horizontal_axis_position()
        if h_pos%360>180:
            move_past = 360-(h_pos%360)+.05
            self.move_horizontal_axis_relative(move_past,wait_til_idle=wait_til_idle)
            # h_busy = True
            # while h_busy:
            #     h_busy = self.is_horizontal_axis_busy()
        if wait_til_idle | (h_pos%360<=180):
            self.home_horizontal_axis(wait_til_idle=wait_til_idle)


    def move_horizontal_axis_relative(self,degrees,useRadians=False,wait_til_idle=True):
        if useRadians:
            unit = Units.ANGLE_RADIANS
        else:
            unit = self.axis_unit
        if self.horizontalLimits:
            proposed_move = self.get_positions()[0]+degrees
            if proposed_move > self.horizontalLimits[1]:
                print('Cannot execute move! would go beyond limit of {}deg'.format(self.horizontalLimits[1]))
            elif proposed_move < self.horizontalLimits[0]:
                print('Cannot execute move! would go beyond limit of {}deg'.format(self.horizontalLimits[0]))
            else:
                self.horizontalAxis.move_relative(degrees,unit,wait_until_idle=wait_til_idle)
        else:
            self.horizontalAxis.move_relative(degrees, unit, wait_until_idle=wait_til_idle)

    def move_vertical_axis_relative(self,degrees,useRadians=False,wait_til_idle=True):
        if useRadians:
            unit = Units.ANGLE_RADIANS
        else:
            unit = self.axis_unit
        if self.verticalLimits:
            proposed_move = self.get_positions()[1] + degrees
            if proposed_move > self.verticalLimits[1]:
                print('Cannot execute move! would go beyond limit of {}deg'.format(self.verticalLimits[1]))
            elif proposed_move < self.verticalLimits[0]:
                print('Cannot execute move! would go beyond limit of {}deg'.format(self.verticalLimits[0]))
            else:
                self.verticalAxis.move_relative(degrees, unit, wait_until_idle=wait_til_idle)
        else:
            self.verticalAxis.move_relative(degrees, unit, wait_until_idle=wait_til_idle)

    def move_both_relative(self,degreeH,degreeV,useRadians=False):
        self.move_horizontal_axis_relative(degreeH,useRadians=useRadians,wait_til_idle=False)
        self.move_vertical_axis_relative(degreeV, useRadians=useRadians,wait_til_idle=False)
        h_status = True
        v_status = True
        while h_status|v_status:
            h_status=self.is_horizontal_axis_busy()
            v_status=self.is_vertical_axis_busy()


    def is_axis_busy(self,axisID=0):
        """
        axisID: 0 == vertical axis, 1 == horizontal axis
        """
        if axisID==0:
            answer = self.is_vertical_axis_busy()
        elif axisID==1:
            answer = self.is_horizontal_axis_busy()
        return answer

    def is_horizontal_axis_busy(self):
        resp = self.horizontalAxis.is_busy()
        return resp

    def is_vertical_axis_busy(self):
        resp = self.verticalAxis.is_busy()
        return resp

    def get_horizontal_axis_position(self,returnRaw=False):
        if returnRaw:
            resp = self.horizontalAxis.get_position(self.axis_unit)
        else:
            resp = self.get_positions()[0]
        return resp

    def get_vertical_axis_position(self,returnRaw=False):
        if returnRaw:
            resp = self.verticalAxis.get_position(self.axis_unit)
        else:
            resp = self.get_positions()[1]
        return resp

    def get_positions(self,return_180_referenced=True):
        """
        returns [horizontal,vertical] in degrees
        """
        h_pos = self.get_horizontal_axis_position(True)
        v_pos = self.get_vertical_axis_position(True)
        if return_180_referenced:
            h_pos = h_pos%360
            v_pos = v_pos%360
            if h_pos > 180:
                h_pos = -1*(360-h_pos)
            if v_pos > 180:
                v_pos = -1*(360-v_pos)
        return [h_pos,v_pos]

    def set_vertical_axis_limits(self,min_degrees,max_degrees):
        self.verticalLimits = [min_degrees,max_degrees]

    def set_horizontal_axis_limits(self,min_degrees,max_degrees):
        self.horizontalLimits = [min_degrees,max_degrees]

    def move_to_spot_relative(self,newH,newV,move=True,printMove=False):
        current_spot = self.get_positions()
        move_dist_h = newH - current_spot[0]
        move_dist_v = newV - current_spot[1]
        if printMove:
            print('will move {}h, {}v'.format(move_dist_h, move_dist_v))
        if move:
            self.move_both_relative(move_dist_h, move_dist_v)

    def move_to_spot_H_relatively(self,newH,move=True,printMove=False):
        current_H = self.get_horizontal_axis_position()
        move_dist_h = newH - current_H
        if printMove:
            print('will move {}h'.format(move_dist_h))
        if move:
            self.move_horizontal_axis_relative(move_dist_h)

    def move_to_spot_V_relatively(self,newV,move=True,printMove=False):
        current_V = self.get_vertical_axis_position()
        move_dist_v = newV - current_V
        if printMove:
            print('will move {}v'.format(move_dist_v))
        if move:
            self.move_vertical_axis_relative(move_dist_v)


    ### Todo: get move params for each axis
    ## todo : set move params for each axis
    ## todo: move absolute for each axis
    ## todo: restrict movement to one direction only?

    






if __name__ == '__main__':
    gimbal = EC_XGRSTDEGimbal('/dev/ttyUSB0')
    # gimbal.voyant_home_horizontal_axis()
    # gimbal.voyant_home_vertical_axis()
    # gimbal.home_both_axes()
    # gimbal.home_vertical_axis()
    # gimbal.home_ho_axis()

